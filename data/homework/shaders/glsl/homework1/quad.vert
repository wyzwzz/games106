#version 450

const vec2 ScreenQuadVertexCoord[4]={ vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0) };
const vec2 ScreenQuadTexCoord[4]={ vec2(0, 0), vec2(1, 0), vec2(0, 1.0), vec2(1.0, 1.0) };

layout(location = 0) out vec2 oTexCoord;
//layout(location = 1) out vec2 oScreenCoord;

void main()
{
    gl_Position = vec4(ScreenQuadVertexCoord[gl_VertexIndex].x, ScreenQuadVertexCoord[gl_VertexIndex].y, 1.0, 1.0);
    oTexCoord = ScreenQuadTexCoord[gl_VertexIndex];
//    oScreenCoord = gl_Position.xy;
}