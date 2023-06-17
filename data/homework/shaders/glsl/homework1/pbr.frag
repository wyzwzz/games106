#version 450

layout(location = 0) in vec3 iWorldPos;
layout(location = 1) in vec3 iFragNormal;
layout(location = 2) in vec2 iUV;
layout(location = 3) in vec3 iFragTangent;

// scene or per frame 0
// per object 1
// per primitive 2

layout(std140, set = 0, binding = 0) uniform UBO{
    mat4 Model;
    mat4 View;
    mat4 Proj;
};

layout(std140, set = 0, binding = 1) uniform PerFrame{
    vec3 CameraPos;
};

// scene
layout(set = 0, binding = 2) uniform samplerCube Irradiance;
layout(set = 0, binding = 3) uniform sampler2D BRDFLUT;
layout(set = 0, binding = 4) uniform samplerCube PrefilteredMap;

//per primitive
layout(set = 2, binding = 1) uniform sampler2D AlbedoMap;
layout(set = 2, binding = 2) uniform sampler2D MetallicRoughnessMap;
layout(set = 2, binding = 3) uniform sampler2D AOMap;
layout(set = 2, binding = 4) uniform sampler2D NormalMap;

void main() {
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
