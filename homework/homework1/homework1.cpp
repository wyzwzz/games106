/*
* Vulkan Example - glTF scene loading and rendering
*
* Copyright (C) 2020-2022 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

/*
 * Shows how to load and display a simple scene from a glTF file
 * Note that this isn't a complete glTF loader and only basic functions are shown here
 * This means no complex materials, no animations, no skins, etc.
 * For details on how glTF 2.0 works, see the official spec at https://github.com/KhronosGroup/glTF/tree/master/specification/2.0
 *
 * Other samples will load models using a dedicated model loader with more features (see base/VulkanglTFModel.hpp)
 *
 * If you are looking for a complete glTF implementation, check out https://github.com/SaschaWillems/Vulkan-glTF-PBR/
 */

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#ifdef VK_USE_PLATFORM_ANDROID_KHR
#define TINYGLTF_ANDROID_LOAD_FROM_ASSETS
#endif
#include "tiny_gltf.h"

#include "vulkanexamplebase.h"

#define ENABLE_VALIDATION false

// Contains everything required to render a glTF model in Vulkan
// This class is heavily simplified (compared to glTF's feature set) but retains the basic glTF structure
class VulkanglTFModel
{
public:
	// The class requires some Vulkan objects so it can create it's own resources
	vks::VulkanDevice* vulkanDevice;
	VkQueue copyQueue;

	// The vertex layout for the samples' model
	struct Vertex {
		glm::vec3 pos;
		glm::vec3 normal;
		glm::vec2 uv;
		glm::vec3 color;
        // skin related
        glm::vec4 jointIndices;
        glm::vec4 jointWeights;
	};

	// Single vertex buffer for all primitives
	struct {
		VkBuffer buffer;
		VkDeviceMemory memory;
	} vertices;

	// Single index buffer for all primitives
	struct {
		int count;
		VkBuffer buffer;
		VkDeviceMemory memory;
	} indices;

	// The following structures roughly represent the glTF scene structure
	// To keep things simple, they only contain those properties that are required for this sample
	struct Node;

	// A primitive contains the data for a single draw call
	struct Primitive {
		uint32_t firstIndex;
		uint32_t indexCount;
		int32_t materialIndex;
	};

	// Contains the node's (optional) geometry and can be made up of an arbitrary number of primitives
	struct Mesh {
		std::vector<Primitive> primitives;
	};

    struct ObjectData{
        vks::Buffer buffer;
        struct Values{
            glm::mat4 model;
        }values;
    };

	// A node represents an object in the glTF scene graph
	struct Node {
		Node* parent;
		std::vector<Node*> children;
		Mesh mesh;
		glm::mat4 matrix;
		~Node() {
			for (auto& child : children) {
				delete child;
			}
		}



        uint32_t index;

        //animation related
        glm::vec3 translation;
        glm::vec3 scale{1.f};
        glm::quat rotation;
        // skin related
        int skin = -1;

        VkDescriptorSet descriptorSet;
        ObjectData objectData;

        glm::mat4 getLocalMatrix() {
            return glm::translate(glm::mat4(1.0f), translation)
                * glm::mat4(rotation)
                * glm::scale(glm::mat4(1.0f), scale)
                * matrix;
        }
	};

    // skin structure
    struct Skin{
        std::string name;
        Node* skeletonRoot = nullptr;
        std::vector<glm::mat4> inverseBindMatrices;
        std::vector<Node*> joints;
        vks::Buffer ssbo;
        VkDescriptorSet descriptorSet;
    };

    // animation related structures
    struct AnimationSampler{
        std::string interpolation;
        std::vector<float> inputs;
        std::vector<glm::vec4> outputsVec4;
    };

    struct AnimationChannel{
        std::string path;
        Node* node;
        uint32_t samplerIndex;
    };

    struct Animation{
        std::string name;
        std::vector<AnimationSampler> samplers;
        std::vector<AnimationChannel> channels;
        float start = std::numeric_limits<float>::max();
        float end = std::numeric_limits<float>::min();
        float currentTime = 0.f;
    };


	// A glTF material stores information in e.g. the texture that is attached to it and colors
	struct Material {
		glm::vec4 baseColorFactor = glm::vec4(1.0f);
		int baseColorTextureIndex = -1;

        int metallicRoughnessTextureIndex = -1;

        int normalTextureIndex = -1;

        int occlusionTextureIndex = -1;

        int emissiveTextureIndex= -1;

        VkDescriptorSet descriptorSet;
	};

	// Contains the texture for a single glTF image
	// Images may be reused by texture objects and are as such separated
	struct Image {
		vks::Texture2D texture;
		// We also store (and create) a descriptor set that's used to access this texture from the fragment shader
		VkDescriptorSet descriptorSet;
	};

	// A glTF texture stores a reference to the image and a sampler
	// In this sample, we are only interested in the image
	struct Texture {
		int32_t imageIndex;
	};

	/*
		Model data
	*/
	std::vector<Image> images;
	std::vector<Texture> textures;
	std::vector<Material> materials;
	std::vector<Node*> nodes;
    std::vector<Skin> skins;
    std::vector<Animation> animations;

    uint32_t activeAnimation = 0;

    uint32_t nodeCount = 0;

	~VulkanglTFModel()
	{
		for (auto node : nodes) {
			delete node;
		}
		// Release all Vulkan resources allocated for the model
		vkDestroyBuffer(vulkanDevice->logicalDevice, vertices.buffer, nullptr);
		vkFreeMemory(vulkanDevice->logicalDevice, vertices.memory, nullptr);
		vkDestroyBuffer(vulkanDevice->logicalDevice, indices.buffer, nullptr);
		vkFreeMemory(vulkanDevice->logicalDevice, indices.memory, nullptr);
		for (Image image : images) {
			vkDestroyImageView(vulkanDevice->logicalDevice, image.texture.view, nullptr);
			vkDestroyImage(vulkanDevice->logicalDevice, image.texture.image, nullptr);
			vkDestroySampler(vulkanDevice->logicalDevice, image.texture.sampler, nullptr);
			vkFreeMemory(vulkanDevice->logicalDevice, image.texture.deviceMemory, nullptr);
		}
	}

	/*
		glTF loading functions

		The following functions take a glTF input model loaded via tinyglTF and convert all required data into our own structure
	*/

	void loadImages(tinygltf::Model& input)
	{
		// Images can be stored inside the glTF (which is the case for the sample model), so instead of directly
		// loading them from disk, we fetch them from the glTF loader and upload the buffers
		images.resize(input.images.size());
		for (size_t i = 0; i < input.images.size(); i++) {
			tinygltf::Image& glTFImage = input.images[i];
			// Get the image data from the glTF loader
			unsigned char* buffer = nullptr;
			VkDeviceSize bufferSize = 0;
			bool deleteBuffer = false;
			// We convert RGB-only images to RGBA, as most devices don't support RGB-formats in Vulkan
			if (glTFImage.component == 3) {
				bufferSize = glTFImage.width * glTFImage.height * 4;
				buffer = new unsigned char[bufferSize];
				unsigned char* rgba = buffer;
				unsigned char* rgb = &glTFImage.image[0];
				for (size_t i = 0; i < glTFImage.width * glTFImage.height; ++i) {
					memcpy(rgba, rgb, sizeof(unsigned char) * 3);
					rgba += 4;
					rgb += 3;
				}
				deleteBuffer = true;
			}
			else {
				buffer = &glTFImage.image[0];
				bufferSize = glTFImage.image.size();
			}
			// Load texture from image buffer
			images[i].texture.fromBuffer(buffer, bufferSize, VK_FORMAT_R8G8B8A8_UNORM, glTFImage.width, glTFImage.height, vulkanDevice, copyQueue);
			if (deleteBuffer) {
				delete[] buffer;
			}
		}
	}

	void loadTextures(tinygltf::Model& input)
	{
		textures.resize(input.textures.size());
		for (size_t i = 0; i < input.textures.size(); i++) {
			textures[i].imageIndex = input.textures[i].source;
		}
	}

	void loadMaterials(tinygltf::Model& input)
	{
		materials.resize(input.materials.size());
		for (size_t i = 0; i < input.materials.size(); i++) {
			// We only read the most basic properties required for our sample
			tinygltf::Material glTFMaterial = input.materials[i];
			// Get the base color factor
			if (glTFMaterial.values.find("baseColorFactor") != glTFMaterial.values.end()) {
				materials[i].baseColorFactor = glm::make_vec4(glTFMaterial.values["baseColorFactor"].ColorFactor().data());
			}
			// Get base color texture index
			if (glTFMaterial.values.find("baseColorTexture") != glTFMaterial.values.end()) {
				materials[i].baseColorTextureIndex = glTFMaterial.values["baseColorTexture"].TextureIndex();
			}
            // todo find texture by access member
            if(glTFMaterial.pbrMetallicRoughness.metallicRoughnessTexture.index != -1){

            }

            if(glTFMaterial.normalTexture.index != -1){

            }

            if(glTFMaterial.occlusionTexture.index != -1){

            }

            if(glTFMaterial.emissiveTexture.index != -1){

            }
		}
	}


    Node* nodeFromIndex(uint32_t index){
        std::function<Node*(Node*, uint32_t)> findNode = [&](Node* parent, uint32_t index)->Node*{
            if(parent->index == index) return parent;
            for(auto& c : parent->children){
                auto ret = findNode(c, index);
                if(ret) return ret;
            }
            return nullptr;
        };
        Node* nodeFound = nullptr;
        for(auto& node : nodes){
            nodeFound = findNode(node, index);
            if(nodeFound) break;
        }
        return nodeFound;
    }

    void loadSkins(tinygltf::Model& input){
        skins.resize(input.skins.size());

        for(size_t i = 0; i < input.skins.size(); i++){
            const auto& glTFSkin = input.skins[i];

            skins[i].name = glTFSkin.name;
            skins[i].skeletonRoot = nodeFromIndex(glTFSkin.skeleton);

            for(int jointIndex : glTFSkin.joints){
                auto node = nodeFromIndex(jointIndex);
                if(node){
                    skins[i].joints.push_back(node);
                }
            }

            if(glTFSkin.inverseBindMatrices > -1){
                const auto& accessor = input.accessors[glTFSkin.inverseBindMatrices];
                const auto& bufferView = input.bufferViews[accessor.bufferView];
                const auto& buffer = input.buffers[bufferView.buffer];
                skins[i].inverseBindMatrices.resize(accessor.count);
                std::memcpy(skins[i].inverseBindMatrices.data(),
                            &buffer.data[accessor.byteOffset + bufferView.byteOffset],
                            accessor.count * sizeof(glm::mat4));

                VK_CHECK_RESULT(vulkanDevice->createBuffer(
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &skins[i].ssbo,
                        sizeof(glm::mat4) * skins[i].inverseBindMatrices.size(),
                        skins[i].inverseBindMatrices.data()
                        ));
                VK_CHECK_RESULT(skins[i].ssbo.map());

            }

        }
    }

    void loadAnimation(tinygltf::Model& input){
        animations.resize(input.animations.size());

        for(size_t i = 0; i < input.animations.size(); i++){
            const auto& glTFAnimation = input.animations[i];
            animations[i].name = glTFAnimation.name;

            animations[i].samplers.resize(glTFAnimation.samplers.size());
            for(size_t j = 0; j < glTFAnimation.samplers.size(); j++){
                const auto& glTFSampler = glTFAnimation.samplers[j];
                auto& dstSampler = animations[i].samplers[j];
                dstSampler.interpolation = glTFSampler.interpolation;

                // read sampler keyframe input time values
                {
                    const auto& accessor = input.accessors[glTFSampler.input];
                    const auto& bufferView = input.bufferViews[accessor.bufferView];
                    const auto& buffer = input.buffers[bufferView.buffer];
                    const void* dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                    const float* buf = static_cast<const float*>(dataPtr);
                    for(size_t index = 0; index < accessor.count; index++){
                        dstSampler.inputs.push_back(buf[index]);
                    }

                    for(auto input : animations[i].samplers[j].inputs){
                        if(input < animations[i].start){
                            animations[i].start = input;
                        }
                        if(input > animations[i].end){
                            animations[i].end = input;
                        }
                    }
                }

                // read sampler keyframe output translate/rotate/scale values
                {
                    const auto& accessor = input.accessors[glTFSampler.output];
                    const auto& bufferView = input.bufferViews[accessor.bufferView];
                    const auto& buffer = input.buffers[bufferView.buffer];
                    const void* dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                    switch (accessor.type){
                        case TINYGLTF_TYPE_VEC3:{
                            const glm::vec3 *buf = static_cast<const glm::vec3 *>(dataPtr);
                            for (size_t index = 0; index < accessor.count; index++)
                            {
                                dstSampler.outputsVec4.push_back(glm::vec4(buf[index], 0.0f));
                            }

                            break;
                        }
                        case TINYGLTF_TYPE_VEC4:{
                            const glm::vec4 *buf = static_cast<const glm::vec4 *>(dataPtr);
                            for (size_t index = 0; index < accessor.count; index++)
                            {
                                dstSampler.outputsVec4.push_back(buf[index]);
                            }

                            break;
                        }
                        default: {
                            std::cout << "unknown type" << std::endl;
                            break;
                        }
                    }
                }
            }

            // channels
            animations[i].channels.resize(glTFAnimation.channels.size());
            for(size_t j = 0; j < glTFAnimation.channels.size(); j++){
                const auto& glTFChannel = glTFAnimation.channels[j];
                auto& dstChannel = animations[i].channels[j];
                dstChannel.path = glTFChannel.target_path;
                dstChannel.samplerIndex = glTFChannel.sampler;
                dstChannel.node = nodeFromIndex(glTFChannel.target_node);
            }
        }
    }


	void loadNode(const tinygltf::Node& inputNode,
                  const tinygltf::Model& input,
                  VulkanglTFModel::Node* parent,
                  uint32_t nodeIndex,
                  std::vector<uint32_t>& indexBuffer,
                  std::vector<VulkanglTFModel::Vertex>& vertexBuffer)
	{
		VulkanglTFModel::Node* node = new VulkanglTFModel::Node{};
		node->matrix = glm::mat4(1.0f);
		node->parent = parent;

        node->index = nodeIndex;
        node->skin = inputNode.skin;

        nodeCount++;

		// Get the local node matrix
		// It's either made up from translation, rotation, scale or a 4x4 matrix
        if (inputNode.translation.size() == 3)
        {
            node->translation = glm::make_vec3(inputNode.translation.data());
        }
        if (inputNode.rotation.size() == 4)
        {
            glm::quat q    = glm::make_quat(inputNode.rotation.data());
            node->rotation = glm::mat4(q);
        }
        if (inputNode.scale.size() == 3)
        {
            node->scale = glm::make_vec3(inputNode.scale.data());
        }
		if (inputNode.matrix.size() == 16) {
			node->matrix = glm::make_mat4x4(inputNode.matrix.data());
		};

		// Load node's children
		if (inputNode.children.size() > 0) {
			for (size_t i = 0; i < inputNode.children.size(); i++) {
				loadNode(input.nodes[inputNode.children[i]], input , node, inputNode.children[i], indexBuffer, vertexBuffer);
			}
		}

		// If the node contains mesh data, we load vertices and indices from the buffers
		// In glTF this is done via accessors and buffer views
		if (inputNode.mesh > -1) {
			const tinygltf::Mesh mesh = input.meshes[inputNode.mesh];
			// Iterate through all primitives of this node's mesh
			for (size_t i = 0; i < mesh.primitives.size(); i++) {
				const tinygltf::Primitive& glTFPrimitive = mesh.primitives[i];
				uint32_t firstIndex = static_cast<uint32_t>(indexBuffer.size());
				uint32_t vertexStart = static_cast<uint32_t>(vertexBuffer.size());
				uint32_t indexCount = 0;
                bool hasSkin = false;
				// Vertices
				{
					const float* positionBuffer = nullptr;
					const float* normalsBuffer = nullptr;
					const float* texCoordsBuffer = nullptr;
					size_t vertexCount = 0;

					// Get buffer data for vertex positions
					if (glTFPrimitive.attributes.find("POSITION") != glTFPrimitive.attributes.end()) {
						const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.attributes.find("POSITION")->second];
						const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
						positionBuffer = reinterpret_cast<const float*>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
						vertexCount = accessor.count;
					}
					// Get buffer data for vertex normals
					if (glTFPrimitive.attributes.find("NORMAL") != glTFPrimitive.attributes.end()) {
						const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.attributes.find("NORMAL")->second];
						const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
						normalsBuffer = reinterpret_cast<const float*>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
					}
					// Get buffer data for vertex texture coordinates
					// glTF supports multiple sets, we only load the first one
					if (glTFPrimitive.attributes.find("TEXCOORD_0") != glTFPrimitive.attributes.end()) {
						const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.attributes.find("TEXCOORD_0")->second];
						const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
						texCoordsBuffer = reinterpret_cast<const float*>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
					}

                    const uint16_t* jointIndicesBuffer = nullptr;
                    const float* jointWeightsBuffer = nullptr;

                    // Get vertex joint indices
                    if(auto it = glTFPrimitive.attributes.find("JOINTS_0"); it != glTFPrimitive.attributes.end()){
                        const auto& accessor = input.accessors[it->second];
                        const auto& view = input.bufferViews[accessor.bufferView];
                        jointIndicesBuffer = reinterpret_cast<const uint16_t*>(&input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]);
                    }

                    // get vertex joint weights
                    if(auto it = glTFPrimitive.attributes.find("WEIGHTS_0"); it != glTFPrimitive.attributes.end()){
                        const auto& accessor = input.accessors[it->second];
                        const auto& view = input.bufferViews[accessor.bufferView];
                        jointWeightsBuffer = reinterpret_cast<const float*>(&input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]);
                    }

                    hasSkin = jointIndicesBuffer && jointWeightsBuffer;

                    for(size_t v = 0; v < vertexCount; v++){
                        auto& vert = vertexBuffer.emplace_back();
                        vert.pos = glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.f);
                        vert.normal = normalsBuffer ? glm::normalize(glm::make_vec3(&normalsBuffer[v * 3])) : glm::vec3(0.f);
                        vert.uv = texCoordsBuffer ? glm::make_vec2(&texCoordsBuffer[v * 2]) : glm::vec2(0.f);
                        vert.color = glm::vec3(1.f);
                        if(hasSkin){
                            vert.jointIndices = glm::vec4(glm::make_vec4(&jointIndicesBuffer[v * 4]));
                            vert.jointWeights = glm::make_vec4(&jointWeightsBuffer[v * 4]);
                        }
                    }

				}
				// Indices
				{
					const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.indices];
					const tinygltf::BufferView& bufferView = input.bufferViews[accessor.bufferView];
					const tinygltf::Buffer& buffer = input.buffers[bufferView.buffer];

					indexCount += static_cast<uint32_t>(accessor.count);

					// glTF supports different component types of indices
					switch (accessor.componentType) {
					case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
						const uint32_t* buf = reinterpret_cast<const uint32_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
						for (size_t index = 0; index < accessor.count; index++) {
							indexBuffer.push_back(buf[index] + vertexStart);
						}
						break;
					}
					case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
						const uint16_t* buf = reinterpret_cast<const uint16_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
						for (size_t index = 0; index < accessor.count; index++) {
							indexBuffer.push_back(buf[index] + vertexStart);
						}
						break;
					}
					case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
						const uint8_t* buf = reinterpret_cast<const uint8_t*>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
						for (size_t index = 0; index < accessor.count; index++) {
							indexBuffer.push_back(buf[index] + vertexStart);
						}
						break;
					}
					default:
						std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
						return;
					}
				}
				Primitive primitive{};
				primitive.firstIndex = firstIndex;
				primitive.indexCount = indexCount;
				primitive.materialIndex = glTFPrimitive.material;
				node->mesh.primitives.push_back(primitive);
			}
		}

		if (parent) {
			parent->children.push_back(node);
		}
		else {
			nodes.push_back(node);
		}
	}

    // glTF vertex skinning functions

    glm::mat4 getNodeMatrix(Node* node){
        auto nodeMatrix = node->getLocalMatrix();
        auto currentParent = node->parent;
        while(currentParent){
            nodeMatrix = currentParent->getLocalMatrix() * nodeMatrix;
            currentParent = currentParent->parent;
        }
        return nodeMatrix;
    }


    void updateJoints(Node* node){
        if(node->skin > -1){
            auto inverseTransform = glm::inverse(getNodeMatrix(node));

            auto& skin = skins[node->skin];
            auto numJoints = skin.joints.size();
            std::vector<glm::mat4> jointMatrices(numJoints);
            for(size_t i = 0; i < numJoints; i++){
                jointMatrices[i] = getNodeMatrix(skin.joints[i]) * skin.inverseBindMatrices[i];

                jointMatrices[i] = inverseTransform * jointMatrices[i];
            }

            skin.ssbo.copyTo(jointMatrices.data(), jointMatrices.size() * sizeof(glm::mat4));

        }

        for(auto* child : node->children){
            updateJoints(child);
        }
    }

    void updateAnimation(float deltaTime){
        if(activeAnimation >= animations.size()){
            std::cerr << "No animation with index " << activeAnimation << std::endl;
            return;
        }
        auto& animation = animations.at(activeAnimation);
        animation.currentTime += deltaTime;
        animation.currentTime = fmod(animation.currentTime, animation.end - animation.start);

        for(auto& channel : animation.channels){
            auto& sampler = animation.samplers[channel.samplerIndex];
            for(size_t i = 0; i < sampler.inputs.size() - 1; i++){
                if(sampler.interpolation != "LINEAR"){
                    std::cerr << "Not supports non-linear interpolations" << std::endl;
                    continue;
                }
                if((animation.currentTime >= sampler.inputs[i])
                && animation.currentTime <= sampler.inputs[i + 1]){
                    float a = (animation.currentTime - sampler.inputs[i])
                            / (sampler.inputs[i + 1] - sampler.inputs[i]);
                    if(channel.path == "translation"){
                        channel.node->translation =
                                glm::mix(sampler.outputsVec4[i],
                                         sampler.outputsVec4[i + 1],
                                         a);
                    }
                    if(channel.path == "rotation"){
                        glm::quat q1;
                        q1.x = sampler.outputsVec4[i].x;
                        q1.y = sampler.outputsVec4[i].y;
                        q1.z = sampler.outputsVec4[i].z;
                        q1.w = sampler.outputsVec4[i].w;

                        glm::quat q2;
                        q2.x = sampler.outputsVec4[i + 1].x;
                        q2.y = sampler.outputsVec4[i + 1].y;
                        q2.z = sampler.outputsVec4[i + 1].z;
                        q2.w = sampler.outputsVec4[i + 1].w;

                        channel.node->rotation =
                                glm::normalize(glm::slerp(q1, q2, a));
                    }
                    if(channel.path == "scale"){
                        channel.node->scale = glm::mix(
                                sampler.outputsVec4[i],
                                sampler.outputsVec4[i + 1],
                                a);
                    }
                }
            }
        }

        for(auto* node : nodes){
            updateJoints(node);
        }
    }

	/*
		glTF rendering functions
	*/

	// Draw a single node including child nodes (if present)
	void drawNode(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, VulkanglTFModel::Node* node)
	{
		if (node->mesh.primitives.size() > 0) {
			// Pass the node's matrix via push constants
			// Traverse the node hierarchy to the top-most parent to get the final matrix of the current node

            int hasSkin;

			// Pass the final matrix to the vertex shader using push constants
            hasSkin = node->skin > -1;
			vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(hasSkin), &hasSkin);
			// Bind ssbo for skin
            if(node->skin > -1)
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                    1, 1, &skins[node->skin].descriptorSet, 0, nullptr);

            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                    3, 1, &node->descriptorSet, 0, nullptr);
            for (VulkanglTFModel::Primitive& primitive : node->mesh.primitives) {
				if (primitive.indexCount > 0) {
					// Get the texture index for this primitive
					VulkanglTFModel::Texture texture = textures[materials[primitive.materialIndex].baseColorTextureIndex];
					// Bind the descriptor for the current primitive's texture
                    //todo bind material's descriptor set
					vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 2, 1, &images[texture.imageIndex].descriptorSet, 0, nullptr);
					vkCmdDrawIndexed(commandBuffer, primitive.indexCount, 1, primitive.firstIndex, 0, 0);
				}
			}
		}
		for (auto& child : node->children) {
			drawNode(commandBuffer, pipelineLayout, child);
		}
	}

	// Draw the glTF scene starting at the top-level-nodes
	void draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout)
	{
		// All vertices and indices are stored in single buffers, so we only need to bind once
		VkDeviceSize offsets[1] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertices.buffer, offsets);
		vkCmdBindIndexBuffer(commandBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
		// Render all nodes at top-level
		for (auto& node : nodes) {
			drawNode(commandBuffer, pipelineLayout, node);
		}
	}

};

class VulkanExample : public VulkanExampleBase
{
public:
	bool wireframe = false;

	VulkanglTFModel glTFModel;

    VulkanglTFModel skybox;

    struct Textures {
//        vks::TextureCubeMap environmentCube;
        // Generated at runtime
        vks::Texture2D lutBrdf;
        vks::TextureCubeMap irradianceCube;
        vks::TextureCubeMap prefilteredCube;

    } textures;

    VkDescriptorSet iblDescriptorSet;

	struct ShaderData {
		vks::Buffer buffer;
		struct Values {
			glm::mat4 projection;
			glm::mat4 model;
			glm::vec4 lightPos = glm::vec4(5.0f, 5.0f, -5.0f, 1.0f);
			glm::vec4 viewPos;
		} values;
	} shaderData;



	struct Pipelines {
		VkPipeline solid;// todo rename to pbr
        VkPipeline sky_view;
        VkPipeline tone_mapping;
		VkPipeline wireframe = VK_NULL_HANDLE;
	} pipelines;

	VkPipelineLayout pipelineLayout;
	VkDescriptorSet descriptorSet;

	struct DescriptorSetLayouts {
		VkDescriptorSetLayout matrices;
		VkDescriptorSetLayout textures;
        VkDescriptorSetLayout jointMatrices;
        VkDescriptorSetLayout objects;
	} descriptorSetLayouts;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "homework1";
		camera.type = Camera::CameraType::lookat;
		camera.flipY = true;
        camera.setPosition(glm::vec3(0.0f, 0.75f, -2.0f));
        camera.setRotation(glm::vec3(0.0f, 0.0f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources
		// Note : Inherited destructor cleans up resources stored in base class
		vkDestroyPipeline(device, pipelines.solid, nullptr);
		if (pipelines.wireframe != VK_NULL_HANDLE) {
			vkDestroyPipeline(device, pipelines.wireframe, nullptr);
		}

		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.matrices, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.textures, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.jointMatrices, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.objects, nullptr);

		shaderData.buffer.destroy();
	}

	virtual void getEnabledFeatures()
	{
		// Fill mode non solid is required for wireframe display
		if (deviceFeatures.fillModeNonSolid) {
			enabledFeatures.fillModeNonSolid = VK_TRUE;
		};
	}

	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[0].color = { { 0.25f, 0.25f, 0.25f, 1.0f } };;
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		const VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
		const VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			renderPassBeginInfo.framebuffer = frameBuffers[i];
			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));
			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);
			// Bind scene matrices descriptor to set 0
            // todo bind ibl resource to set 0
			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, wireframe ? pipelines.wireframe : pipelines.solid);

            glTFModel.draw(drawCmdBuffers[i], pipelineLayout);
			drawUI(drawCmdBuffers[i]);
			vkCmdEndRenderPass(drawCmdBuffers[i]);
			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void loadglTFFile(std::string filename, VulkanglTFModel& glTFModel)
	{
		tinygltf::Model glTFInput;
		tinygltf::TinyGLTF gltfContext;
		std::string error, warning;

		this->device = device;

#if defined(__ANDROID__)
		// On Android all assets are packed with the apk in a compressed form, so we need to open them using the asset manager
		// We let tinygltf handle this, by passing the asset manager of our app
		tinygltf::asset_manager = androidApp->activity->assetManager;
#endif
		bool fileLoaded = gltfContext.LoadASCIIFromFile(&glTFInput, &error, &warning, filename);

		// Pass some Vulkan resources required for setup and rendering to the glTF model loading class
		glTFModel.vulkanDevice = vulkanDevice;
		glTFModel.copyQueue = queue;

		std::vector<uint32_t> indexBuffer;
		std::vector<VulkanglTFModel::Vertex> vertexBuffer;

		if (fileLoaded) {
            // images, materials, textures, skins, animations, these just need load once
            // each node use these by index
			glTFModel.loadImages(glTFInput);
			glTFModel.loadMaterials(glTFInput);
			glTFModel.loadTextures(glTFInput);
            // just consider one scene
			const tinygltf::Scene& scene = glTFInput.scenes[0];
			for (size_t i = 0; i < scene.nodes.size(); i++) {
				const tinygltf::Node& node = glTFInput.nodes[scene.nodes[i]];
				glTFModel.loadNode(node, glTFInput, nullptr, scene.nodes[i], indexBuffer, vertexBuffer);
			}
            glTFModel.loadSkins(glTFInput);
            glTFModel.loadAnimation(glTFInput);
            // calculate initial pose
            for(auto node : glTFModel.nodes){
                glTFModel.updateJoints(node);
            }
		}
		else {
			vks::tools::exitFatal("Could not open the glTF file.\n\nThe file is part of the additional asset pack.\n\nRun \"download_assets.py\" in the repository root to download the latest version.", -1);
			return;
		}

		// Create and upload vertex and index buffer
		// We will be using one single vertex buffer and one single index buffer for the whole glTF scene
		// Primitives (of the glTF model) will then index into these using index offsets

		size_t vertexBufferSize = vertexBuffer.size() * sizeof(VulkanglTFModel::Vertex);
		size_t indexBufferSize = indexBuffer.size() * sizeof(uint32_t);
		glTFModel.indices.count = static_cast<uint32_t>(indexBuffer.size());

		struct StagingBuffer {
			VkBuffer buffer;
			VkDeviceMemory memory;
		} vertexStaging, indexStaging;

		// Create host visible staging buffers (source)
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			vertexBufferSize,
			&vertexStaging.buffer,
			&vertexStaging.memory,
			vertexBuffer.data()));
		// Index data
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			indexBufferSize,
			&indexStaging.buffer,
			&indexStaging.memory,
			indexBuffer.data()));

		// Create device local buffers (target)
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			vertexBufferSize,
			&glTFModel.vertices.buffer,
			&glTFModel.vertices.memory));
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			indexBufferSize,
			&glTFModel.indices.buffer,
			&glTFModel.indices.memory));

		// Copy data from staging buffers (host) do device local buffer (gpu)
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};

		copyRegion.size = vertexBufferSize;
		vkCmdCopyBuffer(
			copyCmd,
			vertexStaging.buffer,
			glTFModel.vertices.buffer,
			1,
			&copyRegion);

		copyRegion.size = indexBufferSize;
		vkCmdCopyBuffer(
			copyCmd,
			indexStaging.buffer,
			glTFModel.indices.buffer,
			1,
			&copyRegion);

		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		// Free staging resources
		vkDestroyBuffer(device, vertexStaging.buffer, nullptr);
		vkFreeMemory(device, vertexStaging.memory, nullptr);
		vkDestroyBuffer(device, indexStaging.buffer, nullptr);
		vkFreeMemory(device, indexStaging.memory, nullptr);
	}

	void loadAssets()
	{
        loadglTFFile(getAssetPath() + "models/buster_drone/busterDrone.gltf", glTFModel);
//        loadglTFFile(getAssetPath() + "models/cube.gltf", skybox);
//        loadglTFFile(getAssetPath() + "models/CesiumMan/glTF/CesiumMan.gltf");
//		loadglTFFile(getAssetPath() + "models/FlightHelmet/glTF/FlightHelmet.gltf");


        //todo create skybox
        {

        }
	}

	void setupDescriptors()
	{
		/*
			This sample uses separate descriptor sets (and layouts) for the matrices and materials (textures)
		*/

		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 + glTFModel.nodeCount),
			// One combined image sampler per model image/texture
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(glTFModel.images.size())),
		    // one ssbo per skin
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<uint32_t>(glTFModel.skins.size()))
            };
		// One set for matrices and one per model image/texture
		const uint32_t maxSetCount = static_cast<uint32_t>(glTFModel.images.size()) + static_cast<uint32_t>(glTFModel.skins.size()) + 1 + glTFModel.nodeCount;
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, maxSetCount);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

		// Descriptor set layout for passing matrices
		VkDescriptorSetLayoutBinding setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(&setLayoutBinding, 1);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.matrices));

        // Descriptor set layout for passing material textures
        setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.textures));

        // Descriptor set layout for passing skin joint matrices
        setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.jointMatrices));

        setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.objects));



        // Pipeline layout using both descriptor sets (set 0 = matrices, set 1 = material)
		std::array<VkDescriptorSetLayout, 4> setLayouts = {
                descriptorSetLayouts.matrices,
                descriptorSetLayouts.jointMatrices,
                descriptorSetLayouts.textures,
                descriptorSetLayouts.objects
        };
		VkPipelineLayoutCreateInfo pipelineLayoutCI= vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));

        // We will use push constants to push the local matrices of a primitive to the vertex shader
		VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(glm::mat4), 0);
		// Push constant ranges are part of the pipeline layout
		pipelineLayoutCI.pushConstantRangeCount = 1;
		pipelineLayoutCI.pPushConstantRanges = &pushConstantRange;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout));

		// Descriptor set for scene matrices
		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.matrices, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
		VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &shaderData.buffer.descriptor);
		vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);


        for(auto node : glTFModel.nodes){

            std::function<void(VulkanglTFModel::Node*)> create_set = [&](VulkanglTFModel::Node* node){

                VK_CHECK_RESULT(vulkanDevice->createBuffer(
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &node->objectData.buffer,
                        sizeof(node->objectData.values)));

                // Map persistent
                VK_CHECK_RESULT(node->objectData.buffer.map());

                auto allocInfo =
                        vks::initializers
                        ::descriptorSetAllocateInfo(descriptorPool,
                                                    &descriptorSetLayouts.objects,
                                                    1);
                VK_CHECK_RESULT(vkAllocateDescriptorSets(device,
                                                         &allocInfo,
                                                         &node->descriptorSet));
                auto writeDescriptorSet = vks::initializers
                ::writeDescriptorSet(node->descriptorSet,
                                     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                     1,
                                     &node->objectData.buffer.descriptor);
                vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);

                for(auto c : node->children){
                    create_set(c);
                }
            };

            create_set(node);

        }

        // Descriptor set for glTF mode skin joint matrices
        for(auto& skin : glTFModel.skins){
            auto allocInfo =
                    vks::initializers
                    ::descriptorSetAllocateInfo(descriptorPool,
                                                &descriptorSetLayouts.jointMatrices,
                                                1);
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device,
                                                     &allocInfo,
                                                     &skin.descriptorSet));
            auto writeDescriptorSet = vks::initializers
                    ::writeDescriptorSet(skin.descriptorSet,
                                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         0,
                                         &skin.ssbo.descriptor);
            vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
        }


        // Descriptor sets for materials
		for (auto& image : glTFModel.images) {
			const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.textures, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &image.descriptorSet));
			VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(image.descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &image.texture.descriptor);
			vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		}
	}

	void preparePipelines()
	{
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentStateCI = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentStateCI);
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		const std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), static_cast<uint32_t>(dynamicStateEnables.size()), 0);
		// Vertex input bindings and attributes
		const std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(VulkanglTFModel::Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
		};
		const std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, pos)),	// Location 0: Position
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, normal)),// Location 1: Normal
			vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, uv)),	// Location 2: Texture coordinates
			vks::initializers::vertexInputAttributeDescription(0, 3, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, color)),	// Location 3: Color
            {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VulkanglTFModel::Vertex, jointIndices)},
            {5, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VulkanglTFModel::Vertex, jointWeights)}
		};
		VkPipelineVertexInputStateCreateInfo vertexInputStateCI = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputStateCI.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputStateCI.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();

		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
			loadShader(getHomeworkShadersPath() + "homework1/mesh.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getHomeworkShadersPath() + "homework1/mesh.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
		};

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass, 0);
		pipelineCI.pVertexInputState = &vertexInputStateCI;
		pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCI.pRasterizationState = &rasterizationStateCI;
		pipelineCI.pColorBlendState = &colorBlendStateCI;
		pipelineCI.pMultisampleState = &multisampleStateCI;
		pipelineCI.pViewportState = &viewportStateCI;
		pipelineCI.pDepthStencilState = &depthStencilStateCI;
		pipelineCI.pDynamicState = &dynamicStateCI;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();

		// Solid rendering pipeline
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.solid));

        // sky view
        // todo reset vertex IA
        rasterizationStateCI.cullMode = VK_CULL_MODE_FRONT_BIT;
        shaderStages[0] = loadShader(getHomeworkShadersPath() + "homework1/cube.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shaderStages[1] = loadShader(getHomeworkShadersPath() + "homework1/sky_demo_proc.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
        depthStencilStateCI.depthWriteEnable = VK_FALSE;
        depthStencilStateCI.depthTestEnable = VK_FALSE;
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.sky_view));

        // tone mapping
        // todo reset vertex IA
        rasterizationStateCI.cullMode = VK_CULL_MODE_BACK_BIT;
        shaderStages[0] = loadShader(getHomeworkShadersPath() + "homework1/quad.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shaderStages[1] = loadShader(getHomeworkShadersPath() + "homework1/tonemapping.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.tone_mapping));

		// Wire frame rendering pipeline
		if (deviceFeatures.fillModeNonSolid) {
			rasterizationStateCI.polygonMode = VK_POLYGON_MODE_LINE;
			rasterizationStateCI.lineWidth = 1.0f;
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.wireframe));
		}
	}

    void generateIrradianceCube(){
        auto tStart = std::chrono::high_resolution_clock::now();

        const VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
        const int32_t dim = 64;
        const uint32_t numMips = static_cast<uint32_t>(floor(log2(dim))) + 1;

        // Pre-filtered cube map
        // Image
        VkImageCreateInfo imageCI = vks::initializers::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = format;
        imageCI.extent.width = dim;
        imageCI.extent.height = dim;
        imageCI.extent.depth = 1;
        imageCI.mipLevels = numMips;
        imageCI.arrayLayers = 6;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageCI.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &textures.irradianceCube.image));
        VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device, textures.irradianceCube.image, &memReqs);
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &textures.irradianceCube.deviceMemory));
        VK_CHECK_RESULT(vkBindImageMemory(device, textures.irradianceCube.image, textures.irradianceCube.deviceMemory, 0));
        // Image view
        VkImageViewCreateInfo viewCI = vks::initializers::imageViewCreateInfo();
        viewCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        viewCI.format = format;
        viewCI.subresourceRange = {};
        viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCI.subresourceRange.levelCount = numMips;
        viewCI.subresourceRange.layerCount = 6;
        viewCI.image = textures.irradianceCube.image;
        VK_CHECK_RESULT(vkCreateImageView(device, &viewCI, nullptr, &textures.irradianceCube.view));
        // Sampler
        VkSamplerCreateInfo samplerCI = vks::initializers::samplerCreateInfo();
        samplerCI.magFilter = VK_FILTER_LINEAR;
        samplerCI.minFilter = VK_FILTER_LINEAR;
        samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCI.minLod = 0.0f;
        samplerCI.maxLod = static_cast<float>(numMips);
        samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK_RESULT(vkCreateSampler(device, &samplerCI, nullptr, &textures.irradianceCube.sampler));

        textures.irradianceCube.descriptor.imageView = textures.irradianceCube.view;
        textures.irradianceCube.descriptor.sampler = textures.irradianceCube.sampler;
        textures.irradianceCube.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        textures.irradianceCube.device = vulkanDevice;

        // FB, Att, RP, Pipe, etc.
        VkAttachmentDescription attDesc = {};
        // Color attachment
        attDesc.format = format;
        attDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        attDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attDesc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

        VkSubpassDescription subpassDescription = {};
        subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDescription.colorAttachmentCount = 1;
        subpassDescription.pColorAttachments = &colorReference;

        // Use subpass dependencies for layout transitions
        std::array<VkSubpassDependency, 2> dependencies;
        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        // Renderpass
        VkRenderPassCreateInfo renderPassCI = vks::initializers::renderPassCreateInfo();
        renderPassCI.attachmentCount = 1;
        renderPassCI.pAttachments = &attDesc;
        renderPassCI.subpassCount = 1;
        renderPassCI.pSubpasses = &subpassDescription;
        renderPassCI.dependencyCount = 2;
        renderPassCI.pDependencies = dependencies.data();
        VkRenderPass renderpass;
        VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCI, nullptr, &renderpass));

        struct {
            VkImage image;
            VkImageView view;
            VkDeviceMemory memory;
            VkFramebuffer framebuffer;
        } offscreen;

        // Offfscreen framebuffer
        {
            // Color attachment
            VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
            imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
            imageCreateInfo.format = format;
            imageCreateInfo.extent.width = dim;
            imageCreateInfo.extent.height = dim;
            imageCreateInfo.extent.depth = 1;
            imageCreateInfo.mipLevels = 1;
            imageCreateInfo.arrayLayers = 1;
            imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VK_CHECK_RESULT(vkCreateImage(device, &imageCreateInfo, nullptr, &offscreen.image));

            VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
            VkMemoryRequirements memReqs;
            vkGetImageMemoryRequirements(device, offscreen.image, &memReqs);
            memAlloc.allocationSize = memReqs.size;
            memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &offscreen.memory));
            VK_CHECK_RESULT(vkBindImageMemory(device, offscreen.image, offscreen.memory, 0));

            VkImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
            colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
            colorImageView.format = format;
            colorImageView.flags = 0;
            colorImageView.subresourceRange = {};
            colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            colorImageView.subresourceRange.baseMipLevel = 0;
            colorImageView.subresourceRange.levelCount = 1;
            colorImageView.subresourceRange.baseArrayLayer = 0;
            colorImageView.subresourceRange.layerCount = 1;
            colorImageView.image = offscreen.image;
            VK_CHECK_RESULT(vkCreateImageView(device, &colorImageView, nullptr, &offscreen.view));

            VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
            fbufCreateInfo.renderPass = renderpass;
            fbufCreateInfo.attachmentCount = 1;
            fbufCreateInfo.pAttachments = &offscreen.view;
            fbufCreateInfo.width = dim;
            fbufCreateInfo.height = dim;
            fbufCreateInfo.layers = 1;
            VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &offscreen.framebuffer));

            VkCommandBuffer layoutCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
            vks::tools::setImageLayout(
                    layoutCmd,
                    offscreen.image,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            vulkanDevice->flushCommandBuffer(layoutCmd, queue, true);
        }

        // Descriptors
        VkDescriptorSetLayout descriptorsetlayout;
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
        };
        VkDescriptorSetLayoutCreateInfo descriptorsetlayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorsetlayoutCI, nullptr, &descriptorsetlayout));

        // Descriptor Pool
//        std::vector<VkDescriptorPoolSize> poolSizes = { vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1) };
//        VkDescriptorPoolCreateInfo descriptorPoolCI = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
//        VkDescriptorPool descriptorpool;
//        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCI, nullptr, &descriptorpool));

        // Descriptor sets
//        VkDescriptorSet descriptorset;
//        VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorpool, &descriptorsetlayout, 1);
//        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorset));
//        VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(descriptorset, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &textures.environmentCube.descriptor);
//        vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);

        // Pipeline layout
        struct PushBlock {
            glm::mat4 mvp;
            // Sampling deltas
            float deltaPhi = (2.0f * float(M_PI)) / 180.0f;
            float deltaTheta = (0.5f * float(M_PI)) / 64.0f;
        } pushBlock;

        VkPipelineLayout pipelinelayout;
        std::vector<VkPushConstantRange> pushConstantRanges = {
                vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(PushBlock), 0),
        };
        VkPipelineLayoutCreateInfo pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorsetlayout, 1);
        pipelineLayoutCI.pushConstantRangeCount = 1;
        pipelineLayoutCI.pPushConstantRanges = pushConstantRanges.data();
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelinelayout));

        // Pipeline
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
        VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
        VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
        VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL);
        VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1);
        VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT);
        std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

        const std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
                vks::initializers::vertexInputBindingDescription(0, sizeof(VulkanglTFModel::Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
        };
        const std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
                vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, pos)),	// Location 0: Position
                vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, normal)),// Location 1: Normal
                vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, uv)),	// Location 2: Texture coordinates
                vks::initializers::vertexInputAttributeDescription(0, 3, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, color)),	// Location 3: Color
                {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VulkanglTFModel::Vertex, jointIndices)},
                {5, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VulkanglTFModel::Vertex, jointWeights)}
        };
        VkPipelineVertexInputStateCreateInfo vertexInputStateCI = vks::initializers::pipelineVertexInputStateCreateInfo();
        vertexInputStateCI.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
        vertexInputStateCI.pVertexBindingDescriptions = vertexInputBindings.data();
        vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
        vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();

        VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelinelayout, renderpass);
        pipelineCI.pInputAssemblyState = &inputAssemblyState;
        pipelineCI.pRasterizationState = &rasterizationState;
        pipelineCI.pColorBlendState = &colorBlendState;
        pipelineCI.pMultisampleState = &multisampleState;
        pipelineCI.pViewportState = &viewportState;
        pipelineCI.pDepthStencilState = &depthStencilState;
        pipelineCI.pDynamicState = &dynamicState;
        pipelineCI.stageCount = 2;
        pipelineCI.pStages = shaderStages.data();
        pipelineCI.renderPass = renderpass;
        pipelineCI.pVertexInputState = &vertexInputStateCI;

        shaderStages[0] = loadShader(getHomeworkShadersPath() + "homework1/cube.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shaderStages[1] = loadShader(getShadersPath() + "homework1/irradiance.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
        VkPipeline pipeline;
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipeline));

        // Render

        VkClearValue clearValues[1];
        clearValues[0].color = { { 0.0f, 0.0f, 0.2f, 0.0f } };//todo ?

        VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
        // Reuse render pass from example pass
        renderPassBeginInfo.renderPass = renderpass;
        renderPassBeginInfo.framebuffer = offscreen.framebuffer;
        renderPassBeginInfo.renderArea.extent.width = dim;
        renderPassBeginInfo.renderArea.extent.height = dim;
        renderPassBeginInfo.clearValueCount = 1;
        renderPassBeginInfo.pClearValues = clearValues;

        std::vector<glm::mat4> matrices = {
                // POSITIVE_X
                glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                // NEGATIVE_X
                glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                // POSITIVE_Y
                glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                // NEGATIVE_Y
                glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                // POSITIVE_Z
                glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                // NEGATIVE_Z
                glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        };

        VkCommandBuffer cmdBuf = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

        VkViewport viewport = vks::initializers::viewport((float)dim, (float)dim, 0.0f, 1.0f);
        VkRect2D scissor = vks::initializers::rect2D(dim, dim, 0, 0);

        vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
        vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = numMips;
        subresourceRange.layerCount = 6;

        // Change image layout for all cubemap faces to transfer destination
        vks::tools::setImageLayout(
                cmdBuf,
                textures.irradianceCube.image,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                subresourceRange);

        for (uint32_t m = 0; m < numMips; m++) {
            for (uint32_t f = 0; f < 6; f++) {
                viewport.width = static_cast<float>(dim * std::pow(0.5f, m));
                viewport.height = static_cast<float>(dim * std::pow(0.5f, m));
                vkCmdSetViewport(cmdBuf, 0, 1, &viewport);

                // Render scene from cube face's point of view
                vkCmdBeginRenderPass(cmdBuf, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

                // Update shader push constant block
                pushBlock.mvp = glm::perspective((float)(M_PI / 2.0), 1.0f, 0.1f, 512.0f) * matrices[f];

                vkCmdPushConstants(cmdBuf, pipelinelayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushBlock), &pushBlock);

                vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
//                vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelinelayout, 0, 1, &descriptorset, 0, NULL);

                // todo
//                skybox.draw(cmdBuf,);

                vkCmdEndRenderPass(cmdBuf);

                vks::tools::setImageLayout(
                        cmdBuf,
                        offscreen.image,
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

                // Copy region for transfer from framebuffer to cube face
                VkImageCopy copyRegion = {};

                copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copyRegion.srcSubresource.baseArrayLayer = 0;
                copyRegion.srcSubresource.mipLevel = 0;
                copyRegion.srcSubresource.layerCount = 1;
                copyRegion.srcOffset = { 0, 0, 0 };

                copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copyRegion.dstSubresource.baseArrayLayer = f;
                copyRegion.dstSubresource.mipLevel = m;
                copyRegion.dstSubresource.layerCount = 1;
                copyRegion.dstOffset = { 0, 0, 0 };

                copyRegion.extent.width = static_cast<uint32_t>(viewport.width);
                copyRegion.extent.height = static_cast<uint32_t>(viewport.height);
                copyRegion.extent.depth = 1;

                vkCmdCopyImage(
                        cmdBuf,
                        offscreen.image,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        textures.irradianceCube.image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1,
                        &copyRegion);

                // Transform framebuffer color attachment back
                vks::tools::setImageLayout(
                        cmdBuf,
                        offscreen.image,
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            }
        }

        vks::tools::setImageLayout(
                cmdBuf,
                textures.irradianceCube.image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                subresourceRange);

        vulkanDevice->flushCommandBuffer(cmdBuf, queue);

        vkDestroyRenderPass(device, renderpass, nullptr);
        vkDestroyFramebuffer(device, offscreen.framebuffer, nullptr);
        vkFreeMemory(device, offscreen.memory, nullptr);
        vkDestroyImageView(device, offscreen.view, nullptr);
        vkDestroyImage(device, offscreen.image, nullptr);
//        vkDestroyDescriptorPool(device, descriptorpool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorsetlayout, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelinelayout, nullptr);

        auto tEnd = std::chrono::high_resolution_clock::now();
        auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        std::cout << "Generating irradiance cube with " << numMips << " mip levels took " << tDiff << " ms" << std::endl;
    }

    void generatePrefilteredCube(){
        auto tStart = std::chrono::high_resolution_clock::now();

        const VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT;
        const int32_t dim = 512;
        const uint32_t numMips = static_cast<uint32_t>(floor(log2(dim))) + 1;

        // Pre-filtered cube map
        // Image
        VkImageCreateInfo imageCI = vks::initializers::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = format;
        imageCI.extent.width = dim;
        imageCI.extent.height = dim;
        imageCI.extent.depth = 1;
        imageCI.mipLevels = numMips;
        imageCI.arrayLayers = 6;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageCI.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &textures.prefilteredCube.image));
        VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device, textures.prefilteredCube.image, &memReqs);
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &textures.prefilteredCube.deviceMemory));
        VK_CHECK_RESULT(vkBindImageMemory(device, textures.prefilteredCube.image, textures.prefilteredCube.deviceMemory, 0));
        // Image view
        VkImageViewCreateInfo viewCI = vks::initializers::imageViewCreateInfo();
        viewCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        viewCI.format = format;
        viewCI.subresourceRange = {};
        viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCI.subresourceRange.levelCount = numMips;
        viewCI.subresourceRange.layerCount = 6;
        viewCI.image = textures.prefilteredCube.image;
        VK_CHECK_RESULT(vkCreateImageView(device, &viewCI, nullptr, &textures.prefilteredCube.view));
        // Sampler
        VkSamplerCreateInfo samplerCI = vks::initializers::samplerCreateInfo();
        samplerCI.magFilter = VK_FILTER_LINEAR;
        samplerCI.minFilter = VK_FILTER_LINEAR;
        samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCI.minLod = 0.0f;
        samplerCI.maxLod = static_cast<float>(numMips);
        samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK_RESULT(vkCreateSampler(device, &samplerCI, nullptr, &textures.prefilteredCube.sampler));

        textures.prefilteredCube.descriptor.imageView = textures.prefilteredCube.view;
        textures.prefilteredCube.descriptor.sampler = textures.prefilteredCube.sampler;
        textures.prefilteredCube.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        textures.prefilteredCube.device = vulkanDevice;

        // FB, Att, RP, Pipe, etc.
        VkAttachmentDescription attDesc = {};
        // Color attachment
        attDesc.format = format;
        attDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        attDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attDesc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

        VkSubpassDescription subpassDescription = {};
        subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDescription.colorAttachmentCount = 1;
        subpassDescription.pColorAttachments = &colorReference;

        // Use subpass dependencies for layout transitions
        std::array<VkSubpassDependency, 2> dependencies;
        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        // Renderpass
        VkRenderPassCreateInfo renderPassCI = vks::initializers::renderPassCreateInfo();
        renderPassCI.attachmentCount = 1;
        renderPassCI.pAttachments = &attDesc;
        renderPassCI.subpassCount = 1;
        renderPassCI.pSubpasses = &subpassDescription;
        renderPassCI.dependencyCount = 2;
        renderPassCI.pDependencies = dependencies.data();
        VkRenderPass renderpass;
        VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCI, nullptr, &renderpass));

        struct {
            VkImage image;
            VkImageView view;
            VkDeviceMemory memory;
            VkFramebuffer framebuffer;
        } offscreen;

        // Offfscreen framebuffer
        {
            // Color attachment
            VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
            imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
            imageCreateInfo.format = format;
            imageCreateInfo.extent.width = dim;
            imageCreateInfo.extent.height = dim;
            imageCreateInfo.extent.depth = 1;
            imageCreateInfo.mipLevels = 1;
            imageCreateInfo.arrayLayers = 1;
            imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VK_CHECK_RESULT(vkCreateImage(device, &imageCreateInfo, nullptr, &offscreen.image));

            VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
            VkMemoryRequirements memReqs;
            vkGetImageMemoryRequirements(device, offscreen.image, &memReqs);
            memAlloc.allocationSize = memReqs.size;
            memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &offscreen.memory));
            VK_CHECK_RESULT(vkBindImageMemory(device, offscreen.image, offscreen.memory, 0));

            VkImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
            colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
            colorImageView.format = format;
            colorImageView.flags = 0;
            colorImageView.subresourceRange = {};
            colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            colorImageView.subresourceRange.baseMipLevel = 0;
            colorImageView.subresourceRange.levelCount = 1;
            colorImageView.subresourceRange.baseArrayLayer = 0;
            colorImageView.subresourceRange.layerCount = 1;
            colorImageView.image = offscreen.image;
            VK_CHECK_RESULT(vkCreateImageView(device, &colorImageView, nullptr, &offscreen.view));

            VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
            fbufCreateInfo.renderPass = renderpass;
            fbufCreateInfo.attachmentCount = 1;
            fbufCreateInfo.pAttachments = &offscreen.view;
            fbufCreateInfo.width = dim;
            fbufCreateInfo.height = dim;
            fbufCreateInfo.layers = 1;
            VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &offscreen.framebuffer));

            VkCommandBuffer layoutCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
            vks::tools::setImageLayout(
                    layoutCmd,
                    offscreen.image,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            vulkanDevice->flushCommandBuffer(layoutCmd, queue, true);
        }

        // Descriptors
        VkDescriptorSetLayout descriptorsetlayout;
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
        };
        VkDescriptorSetLayoutCreateInfo descriptorsetlayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorsetlayoutCI, nullptr, &descriptorsetlayout));

        // Descriptor Pool
//        std::vector<VkDescriptorPoolSize> poolSizes = { vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1) };
//        VkDescriptorPoolCreateInfo descriptorPoolCI = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
//        VkDescriptorPool descriptorpool;
//        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCI, nullptr, &descriptorpool));

        // Descriptor sets
//        VkDescriptorSet descriptorset;
//        VkDescriptorSetAllocateInfo allocInfo =	vks::initializers::descriptorSetAllocateInfo(descriptorpool, &descriptorsetlayout, 1);
//        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorset));
//        VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(descriptorset, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &textures.environmentCube.descriptor);
//        vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);

        // Pipeline layout
        struct PushBlock {
            glm::mat4 mvp;
            float roughness;
            uint32_t numSamples = 32u;
        } pushBlock;

        VkPipelineLayout pipelinelayout;
        std::vector<VkPushConstantRange> pushConstantRanges = {
                vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(PushBlock), 0),
        };
        VkPipelineLayoutCreateInfo pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorsetlayout, 1);
        pipelineLayoutCI.pushConstantRangeCount = 1;
        pipelineLayoutCI.pPushConstantRanges = pushConstantRanges.data();
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelinelayout));

        // Pipeline
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
        VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
        VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
        VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL);
        VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1);
        VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT);
        std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

        VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelinelayout, renderpass);
        pipelineCI.pInputAssemblyState = &inputAssemblyState;
        pipelineCI.pRasterizationState = &rasterizationState;
        pipelineCI.pColorBlendState = &colorBlendState;
        pipelineCI.pMultisampleState = &multisampleState;
        pipelineCI.pViewportState = &viewportState;
        pipelineCI.pDepthStencilState = &depthStencilState;
        pipelineCI.pDynamicState = &dynamicState;
        pipelineCI.stageCount = 2;
        pipelineCI.pStages = shaderStages.data();
        pipelineCI.renderPass = renderpass;
        //todo
//        pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal, vkglTF::VertexComponent::UV });

        shaderStages[0] = loadShader(getShadersPath() + "homework1/cube.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shaderStages[1] = loadShader(getShadersPath() + "homework1/prefilter.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
        VkPipeline pipeline;
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipeline));

        // Render

        VkClearValue clearValues[1];
        clearValues[0].color = { { 0.0f, 0.0f, 0.2f, 0.0f } };

        VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
        // Reuse render pass from example pass
        renderPassBeginInfo.renderPass = renderpass;
        renderPassBeginInfo.framebuffer = offscreen.framebuffer;
        renderPassBeginInfo.renderArea.extent.width = dim;
        renderPassBeginInfo.renderArea.extent.height = dim;
        renderPassBeginInfo.clearValueCount = 1;
        renderPassBeginInfo.pClearValues = clearValues;

        std::vector<glm::mat4> matrices = {
                // POSITIVE_X
                glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                // NEGATIVE_X
                glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                // POSITIVE_Y
                glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                // NEGATIVE_Y
                glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                // POSITIVE_Z
                glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                // NEGATIVE_Z
                glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        };

        VkCommandBuffer cmdBuf = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

        VkViewport viewport = vks::initializers::viewport((float)dim, (float)dim, 0.0f, 1.0f);
        VkRect2D scissor = vks::initializers::rect2D(dim, dim, 0, 0);

        vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
        vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = numMips;
        subresourceRange.layerCount = 6;

        // Change image layout for all cubemap faces to transfer destination
        vks::tools::setImageLayout(
                cmdBuf,
                textures.prefilteredCube.image,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                subresourceRange);

        for (uint32_t m = 0; m < numMips; m++) {
            pushBlock.roughness = (float)m / (float)(numMips - 1);
            for (uint32_t f = 0; f < 6; f++) {
                viewport.width = static_cast<float>(dim * std::pow(0.5f, m));
                viewport.height = static_cast<float>(dim * std::pow(0.5f, m));
                vkCmdSetViewport(cmdBuf, 0, 1, &viewport);

                // Render scene from cube face's point of view
                vkCmdBeginRenderPass(cmdBuf, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

                // Update shader push constant block
                pushBlock.mvp = glm::perspective((float)(M_PI / 2.0), 1.0f, 0.1f, 512.0f) * matrices[f];

                vkCmdPushConstants(cmdBuf, pipelinelayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushBlock), &pushBlock);

                vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
//                vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelinelayout, 0, 1, &descriptorset, 0, NULL);

                //todo
//                models.skybox.draw(cmdBuf);

                vkCmdEndRenderPass(cmdBuf);

                vks::tools::setImageLayout(
                        cmdBuf,
                        offscreen.image,
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

                // Copy region for transfer from framebuffer to cube face
                VkImageCopy copyRegion = {};

                copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copyRegion.srcSubresource.baseArrayLayer = 0;
                copyRegion.srcSubresource.mipLevel = 0;
                copyRegion.srcSubresource.layerCount = 1;
                copyRegion.srcOffset = { 0, 0, 0 };

                copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copyRegion.dstSubresource.baseArrayLayer = f;
                copyRegion.dstSubresource.mipLevel = m;
                copyRegion.dstSubresource.layerCount = 1;
                copyRegion.dstOffset = { 0, 0, 0 };

                copyRegion.extent.width = static_cast<uint32_t>(viewport.width);
                copyRegion.extent.height = static_cast<uint32_t>(viewport.height);
                copyRegion.extent.depth = 1;

                vkCmdCopyImage(
                        cmdBuf,
                        offscreen.image,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        textures.prefilteredCube.image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1,
                        &copyRegion);

                // Transform framebuffer color attachment back
                vks::tools::setImageLayout(
                        cmdBuf,
                        offscreen.image,
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            }
        }

        vks::tools::setImageLayout(
                cmdBuf,
                textures.prefilteredCube.image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                subresourceRange);

        vulkanDevice->flushCommandBuffer(cmdBuf, queue);

        vkDestroyRenderPass(device, renderpass, nullptr);
        vkDestroyFramebuffer(device, offscreen.framebuffer, nullptr);
        vkFreeMemory(device, offscreen.memory, nullptr);
        vkDestroyImageView(device, offscreen.view, nullptr);
        vkDestroyImage(device, offscreen.image, nullptr);
//        vkDestroyDescriptorPool(device, descriptorpool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorsetlayout, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelinelayout, nullptr);

        auto tEnd = std::chrono::high_resolution_clock::now();
        auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        std::cout << "Generating pre-filtered enivornment cube with " << numMips << " mip levels took " << tDiff << " ms" << std::endl;
    }

    void generateBRDFLUT(){
        auto tStart = std::chrono::high_resolution_clock::now();

        const VkFormat format = VK_FORMAT_R16G16_SFLOAT;	// R16G16 is supported pretty much everywhere
        const int32_t dim = 512;

        // Image
        VkImageCreateInfo imageCI = vks::initializers::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = format;
        imageCI.extent.width = dim;
        imageCI.extent.height = dim;
        imageCI.extent.depth = 1;
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &textures.lutBrdf.image));
        VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device, textures.lutBrdf.image, &memReqs);
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &textures.lutBrdf.deviceMemory));
        VK_CHECK_RESULT(vkBindImageMemory(device, textures.lutBrdf.image, textures.lutBrdf.deviceMemory, 0));
        // Image view
        VkImageViewCreateInfo viewCI = vks::initializers::imageViewCreateInfo();
        viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCI.format = format;
        viewCI.subresourceRange = {};
        viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCI.subresourceRange.levelCount = 1;
        viewCI.subresourceRange.layerCount = 1;
        viewCI.image = textures.lutBrdf.image;
        VK_CHECK_RESULT(vkCreateImageView(device, &viewCI, nullptr, &textures.lutBrdf.view));
        // Sampler
        VkSamplerCreateInfo samplerCI = vks::initializers::samplerCreateInfo();
        samplerCI.magFilter = VK_FILTER_LINEAR;
        samplerCI.minFilter = VK_FILTER_LINEAR;
        samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCI.minLod = 0.0f;
        samplerCI.maxLod = 1.0f;
        samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK_RESULT(vkCreateSampler(device, &samplerCI, nullptr, &textures.lutBrdf.sampler));

        textures.lutBrdf.descriptor.imageView = textures.lutBrdf.view;
        textures.lutBrdf.descriptor.sampler = textures.lutBrdf.sampler;
        textures.lutBrdf.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        textures.lutBrdf.device = vulkanDevice;

        // FB, Att, RP, Pipe, etc.
        VkAttachmentDescription attDesc = {};
        // Color attachment
        attDesc.format = format;
        attDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        attDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attDesc.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

        VkSubpassDescription subpassDescription = {};
        subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDescription.colorAttachmentCount = 1;
        subpassDescription.pColorAttachments = &colorReference;

        // Use subpass dependencies for layout transitions
        std::array<VkSubpassDependency, 2> dependencies;
        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        // Create the actual renderpass
        VkRenderPassCreateInfo renderPassCI = vks::initializers::renderPassCreateInfo();
        renderPassCI.attachmentCount = 1;
        renderPassCI.pAttachments = &attDesc;
        renderPassCI.subpassCount = 1;
        renderPassCI.pSubpasses = &subpassDescription;
        renderPassCI.dependencyCount = 2;
        renderPassCI.pDependencies = dependencies.data();

        VkRenderPass renderpass;
        VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCI, nullptr, &renderpass));

        VkFramebufferCreateInfo framebufferCI = vks::initializers::framebufferCreateInfo();
        framebufferCI.renderPass = renderpass;
        framebufferCI.attachmentCount = 1;
        framebufferCI.pAttachments = &textures.lutBrdf.view;
        framebufferCI.width = dim;
        framebufferCI.height = dim;
        framebufferCI.layers = 1;

        VkFramebuffer framebuffer;
        VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCI, nullptr, &framebuffer));

        // Descriptors
        VkDescriptorSetLayout descriptorsetlayout;
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {};
        VkDescriptorSetLayoutCreateInfo descriptorsetlayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorsetlayoutCI, nullptr, &descriptorsetlayout));

        // Descriptor Pool
        std::vector<VkDescriptorPoolSize> poolSizes = { vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1) };
        VkDescriptorPoolCreateInfo descriptorPoolCI = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
        VkDescriptorPool descriptorpool;
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCI, nullptr, &descriptorpool));

        // Descriptor sets
        VkDescriptorSet descriptorset;
        VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorpool, &descriptorsetlayout, 1);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorset));

        // Pipeline layout
        VkPipelineLayout pipelinelayout;
        VkPipelineLayoutCreateInfo pipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorsetlayout, 1);
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelinelayout));

        // Pipeline
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
        VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
        VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
        VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL);
        VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1);
        VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT);
        std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
        VkPipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

        VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelinelayout, renderpass);
        pipelineCI.pInputAssemblyState = &inputAssemblyState;
        pipelineCI.pRasterizationState = &rasterizationState;
        pipelineCI.pColorBlendState = &colorBlendState;
        pipelineCI.pMultisampleState = &multisampleState;
        pipelineCI.pViewportState = &viewportState;
        pipelineCI.pDepthStencilState = &depthStencilState;
        pipelineCI.pDynamicState = &dynamicState;
        pipelineCI.stageCount = 2;
        pipelineCI.pStages = shaderStages.data();
        pipelineCI.pVertexInputState = &emptyInputState;

        // Look-up-table (from BRDF) pipeline
        shaderStages[0] = loadShader(getShadersPath() + "homework1/quad.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shaderStages[1] = loadShader(getShadersPath() + "homework1/brdf.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
        VkPipeline pipeline;
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipeline));

        // Render
        VkClearValue clearValues[1];
        clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };

        VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderpass;
        renderPassBeginInfo.renderArea.extent.width = dim;
        renderPassBeginInfo.renderArea.extent.height = dim;
        renderPassBeginInfo.clearValueCount = 1;
        renderPassBeginInfo.pClearValues = clearValues;
        renderPassBeginInfo.framebuffer = framebuffer;

        VkCommandBuffer cmdBuf = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        vkCmdBeginRenderPass(cmdBuf, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        VkViewport viewport = vks::initializers::viewport((float)dim, (float)dim, 0.0f, 1.0f);
        VkRect2D scissor = vks::initializers::rect2D(dim, dim, 0, 0);
        vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
        vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdDraw(cmdBuf, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmdBuf);
        vulkanDevice->flushCommandBuffer(cmdBuf, queue);

        vkQueueWaitIdle(queue);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelinelayout, nullptr);
        vkDestroyRenderPass(device, renderpass, nullptr);
        vkDestroyFramebuffer(device, framebuffer, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorsetlayout, nullptr);
        vkDestroyDescriptorPool(device, descriptorpool, nullptr);

        auto tEnd = std::chrono::high_resolution_clock::now();
        auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        std::cout << "Generating BRDF LUT took " << tDiff << " ms" << std::endl;
    }

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Vertex shader uniform buffer block
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&shaderData.buffer,
			sizeof(shaderData.values)));

		// Map persistent
		VK_CHECK_RESULT(shaderData.buffer.map());

		updateUniformBuffers(true);
	}

	void updateUniformBuffers(bool first = false)
	{
		shaderData.values.projection = camera.matrices.perspective;
		shaderData.values.model = camera.matrices.view;
		shaderData.values.viewPos = camera.viewPos;
		memcpy(shaderData.buffer.mapped, &shaderData.values, sizeof(shaderData.values));

        if(first) return;

        for(auto node : glTFModel.nodes){
            std::function<void(VulkanglTFModel::Node*)> update_model = [&](VulkanglTFModel::Node* node){
                glm::mat4 nodeMatrix = node->getLocalMatrix() * node->matrix;
                VulkanglTFModel::Node *currentParent = node->parent;
                while (currentParent)
                {
                    nodeMatrix    = currentParent->getLocalMatrix() * nodeMatrix;
                    currentParent = currentParent->parent;
                }
                node->objectData.values.model =  nodeMatrix;
                memcpy(node->objectData.buffer.mapped, &node->objectData.values.model, sizeof(node->objectData.values));

                for(auto c : node->children){
                    update_model(c);
                }
            };
            update_model(node);
        }
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		prepareUniformBuffers();
        setupDescriptors();
		preparePipelines();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		renderFrame();
		if (camera.updated) {
			updateUniformBuffers();
		}

        if(!paused){
            glTFModel.updateAnimation(frameTimer);
        }
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
	{
		if (overlay->header("Settings")) {
			if (overlay->checkBox("Wireframe", &wireframe)) {
				buildCommandBuffers();
			}
		}
	}
};

VULKAN_EXAMPLE_MAIN()
