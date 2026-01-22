#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#include <GL/glew.h>

#include <string_view>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <cmath>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/string_cast.hpp>

#include "obj_parser.hpp"
#include "stb_image.h"

std::string to_string(std::string_view str)
{
    return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message)
{
    throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error)
{
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char*>(glewGetErrorString(error)));
}


const char vertex_shader_bottom[] =
R"(#version 330 core

const float W = 7873, D = 5280;

const vec3 vertices[4] = vec3[](
    vec3(-1, 0, -D / W),
    vec3(1, 0, -D / W),
    vec3(-1, 0, D / W),
    vec3(1, 0, D / W)
);

const vec2 texcoords[4] = vec2[](
    vec2(0.0, 0.0),
    vec2( 1.0, 0.0),
    vec2(0.0,  1.0),
    vec2( 1.0,  1.0)
);

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 texcoord;
out vec3 position;

void main()
{
    position = (model * vec4(vertices[gl_VertexID], 1.0)).xyz;
    gl_Position = projection * view * vec4(position, 1.0);
    texcoord = texcoords[gl_VertexID];
}
)";

const char fragment_shader_bottom[] =
R"(#version 330 core

in vec2 texcoord;
in vec3 position;
uniform sampler2D bottom_texture;
uniform sampler2D caustics_texture;
uniform vec3 light_direction;
uniform vec3 camera_position;

const float water_refractive_index = 1.333;

layout (location = 0) out vec4 out_color;

const vec3 ambient_light_color = vec3(1.2);
const vec3 light_color = vec3(1.0);

void main()
{
    vec3 norm = vec3(0.0, 1.0, 0.0);
    vec3 refracted_light_direction = -refract(normalize(light_direction), norm, 1.0/water_refractive_index);
    //refracted_light_direction = -normalize(light_direction);
    
    vec3 albedo = texture(bottom_texture, texcoord).xyz;
    float caustics = texture(caustics_texture, texcoord).r;

    float cosine = dot(norm, refracted_light_direction);
    float light_factor = max(0.0, cosine);

    vec3 ambient = ambient_light_color * albedo;
    vec3 diffuse = light_color * albedo * light_factor;

    out_color = vec4(ambient + diffuse * caustics, 1.0);
}
)";


const char vertex_shader_water[] =
R"(#version 330 core

const float water_y = 0.5;
const float W = 7873, D = 5280;
const int GRID = 1024;

const float x_min = -1;
const float z_min = -D / W;
const float x_max = 1;
const float z_max = D / W;

const vec3 vertices[4] = vec3[](
    vec3(x_min, 0, z_min),
    vec3(x_max, 0, z_min),
    vec3(x_min, 0, z_max),
    vec3(x_max, 0, z_max)
);

const vec2 texcoords[4] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(0.0,  1.0),
    vec2( 1.0,  1.0)
);

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;


float h(float x, float z)
{
    x *= 5;
    z *= 5;
    return
        (0.035 * sin(1.4 * x + 0.9 * time) +
        0.025 * sin(1.1 * z + 1.2 * time + 1.7) +
        0.020 * sin(0.8 * (x + z) + 0.6 * time) +
        0.015 * sin(1.6 * (x - 0.7 * z) + 1.4 * time) +
        0.010 * sin(3.5 * x + 2.9 * z + 2.1 * time) +
        0.008 * sin(5.1 * x - 4.7 * z + 3.3 * time)) * 0.5;
}

float dhdx(float x, float z)
{
    x *= 5;
    z *= 5;

    return 0.5 * 5 * (
        0.035 * 1.4 * cos(1.4 * x + 0.9 * time) +
        0.020 * 0.8 * cos(0.8 * (x + z) + 0.6 * time) +
        0.015 * 1.6 * cos(1.6 * (x - 0.7 * z) + 1.4 * time) +
        0.010 * 3.5 * cos(3.5 * x + 2.9 * z + 2.1 * time) +
        0.008 * 5.1 * cos(5.1 * x - 4.7 * z + 3.3 * time)
    );
}

float dhdz(float x, float z)
{
    x *= 5;
    z *= 5;

    return 0.5 * 5 * (
        0.025 * 1.1 * cos(1.1 * z + 1.2 * time + 1.7) +
        0.020 * 0.8 * cos(0.8 * (x + z) + 0.6 * time) +
        0.015 * (-1.6 * 0.7) * cos(1.6 * (x - 0.7 * z) + 1.4 * time) +
        0.010 * 2.9 * cos(3.5 * x + 2.9 * z + 2.1 * time) +
        0.008 * (-4.7) * cos(5.1 * x - 4.7 * z + 3.3 * time)
    );
}

out vec3 normal;
out vec3 world_position;

void main()
{
    int xi = gl_VertexID % GRID;
    int zi = gl_VertexID / GRID;
    float x = x_min + 2.0 * xi / GRID;
    float z = z_min + 2.0 * D / W * zi / GRID;
    float y = water_y + h(x, z);
    normal = normalize(vec3(-dhdx(x, z), 1.0, -dhdz(x, z)));
    world_position = vec3(x, y, z);
    vec3 position = (model * vec4(world_position, 1.0)).xyz;
    gl_Position = projection * view * vec4(position, 1.0);
}
)";

const char fragment_shader_water[] =
R"(#version 330 core

const float W = 7873, D = 5280;

const float x_min = -1;
const float z_min = -D / W;
const float x_max = 1;
const float z_max = D / W;

const vec3 vertices[4] = vec3[](
    vec3(x_min, 0, z_min),
    vec3(x_max, 0, z_min),
    vec3(x_min, 0, z_max),
    vec3(x_max, 0, z_max)
);

const float water_refractive_index = 1.333;

uniform vec3 light_direction;
uniform vec3 camera_position;
uniform vec3 bottom_world_box[4];
uniform sampler2D bottom_texture;
uniform sampler2D caustics_texture;
uniform sampler2D environment_texture;

in vec3 normal;
in vec3 world_position;

layout (location = 0) out vec4 out_color;

const float diffuse_albedo = 0.3;

const float PI = 3.141592653589793;

vec3 environment_map(vec3 dir) {
    float x = atan(dir.z, dir.x) / PI * 0.5 + 0.5;
    float y = -atan(dir.y, length(dir.xz)) / PI + 0.5;

    return texture(environment_texture, vec2(x, y)).xyz;
}

const vec3 ambient_light_color = vec3(1.2);
const vec3 light_color = vec3(1.0);

const vec3 water_ambient_light_color = vec3(1.0);
const vec3 water_light_color = vec3(0.1);
const float specular_power = 128;
const vec3 specular_light_color = vec3(1.5);

void main()
{
    vec3 norm = normalize(normal);
    vec3 camera_dir = normalize(world_position - camera_position);

    vec3 reflected_dir = reflect(camera_dir, norm);
    vec3 reflected_color = environment_map(reflected_dir);


    vec3 refracted_dir = refract(camera_dir, norm, 1.0 / water_refractive_index);
    vec3 refracted_color;
    if (refracted_dir.y == 0)
        refracted_color = reflected_color;
    else {
        vec3 p = world_position - refracted_dir * (world_position.y / refracted_dir.y);
        if (x_min <= p.x && p.x <= x_max && z_min <= p.z && p.z <= z_max) {
            vec2 texcoord = vec2((p.x - x_min) / (x_max - x_min), (p.z - z_min) / (z_max - z_min));

            vec3 norm1 = vec3(0.0, 1.0, 0.0);
            vec3 refracted_light_direction = -refract(normalize(light_direction), norm1, 1.0/water_refractive_index);
    
            vec3 albedo = texture(bottom_texture, texcoord).xyz;
            float caustics = texture(caustics_texture, texcoord).r;

            float cosine = dot(norm1, refracted_light_direction);
            float light_factor = max(0.0, cosine);

            vec3 ambient = ambient_light_color * albedo;
            vec3 diffuse = light_color * albedo * light_factor;

            refracted_color = ambient + diffuse * caustics;
        }
        else {
            refracted_color = environment_map(refracted_dir);
        }
    }

    float R0 = pow((1.0 - water_refractive_index) / (1.0 + water_refractive_index), 2);
    float fresnel = R0 + (1.0 - R0) * pow(1.0 - max(dot(-camera_dir, norm), 0.0), 5.0);

    vec3 albedo = mix(refracted_color, reflected_color, fresnel);

    vec3 view_direction = -camera_dir;
    vec3 light_dir = -light_direction;

    float cosine = dot(norm, light_dir);
    float light_factor = max(0.0, cosine);
    vec3 reflected_direction = 2.0 * norm * cosine - light_dir;

    vec3 ambient = water_ambient_light_color * albedo;
    vec3 diffuse = water_light_color * albedo * light_factor;
    vec3 specular = specular_light_color * pow(max(dot(reflected_direction, view_direction), 0.0), specular_power);

    out_color = vec4(ambient + diffuse + specular, 1.0);
}
)";


const char vertex_shader_caustics[] =
R"(#version 330 core

const float water_refractive_index = 1.333;
const float water_y = 0.5;
const float W = 7873, D = 5280;
const int GRID = 1024;

const float x_min = -1;
const float z_min = -D / W;
const float x_max = 1;
const float z_max = D / W;

const vec3 vertices[4] = vec3[](
    vec3(x_min, 0, z_min),
    vec3(x_max, 0, z_min),
    vec3(x_min, 0, z_max),
    vec3(x_max, 0, z_max)
);

const vec2 texcoords[4] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(0.0,  1.0),
    vec2(1.0,  1.0)
);

uniform float time;
uniform vec3 light_direction;


float h(float x, float z)
{
    x *= 5;
    z *= 5;
    return
        (0.035 * sin(1.4 * x + 0.9 * time) +
        0.025 * sin(1.1 * z + 1.2 * time + 1.7) +
        0.020 * sin(0.8 * (x + z) + 0.6 * time) +
        0.015 * sin(1.6 * (x - 0.7 * z) + 1.4 * time) +
        0.010 * sin(3.5 * x + 2.9 * z + 2.1 * time) +
        0.008 * sin(5.1 * x - 4.7 * z + 3.3 * time)) * 0.5;
}

float dhdx(float x, float z)
{
    x *= 5;
    z *= 5;

    return 0.5 * 5 * (
        0.035 * 1.4 * cos(1.4 * x + 0.9 * time) +
        0.020 * 0.8 * cos(0.8 * (x + z) + 0.6 * time) +
        0.015 * 1.6 * cos(1.6 * (x - 0.7 * z) + 1.4 * time) +
        0.010 * 3.5 * cos(3.5 * x + 2.9 * z + 2.1 * time) +
        0.008 * 5.1 * cos(5.1 * x - 4.7 * z + 3.3 * time)
    );
}

float dhdz(float x, float z)
{
    x *= 5;
    z *= 5;

    return 0.5 * 5 * (
        0.025 * 1.1 * cos(1.1 * z + 1.2 * time + 1.7) +
        0.020 * 0.8 * cos(0.8 * (x + z) + 0.6 * time) +
        0.015 * (-1.6 * 0.7) * cos(1.6 * (x - 0.7 * z) + 1.4 * time) +
        0.010 * 2.9 * cos(3.5 * x + 2.9 * z + 2.1 * time) +
        0.008 * (-4.7) * cos(5.1 * x - 4.7 * z + 3.3 * time)
    );
}

void main()
{
    int xi = gl_VertexID % GRID;
    int zi = gl_VertexID / GRID;
    float x = x_min + 2.0 * xi / GRID;
    float z = z_min + 2.0 * D / W * zi / GRID;
    float y = water_y + h(x, z);
    vec3 normal = normalize(vec3(-dhdx(x, z), 1.0, -dhdz(x, z)));
    vec3 world_position = vec3(x, y, z);

    vec3 refracted_light_dir = refract(light_direction, normal, 1.0 / water_refractive_index);

    vec3 p = world_position - refracted_light_dir * (world_position.y / refracted_light_dir.y);

    mat4 projection = mat4(
        x_max, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 2 * z_max, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    gl_Position = projection * vec4(p, 1.0);
}
)";

const char fragment_shader_caustics[] = 
R"(#version 330 core

layout (location = 0) out vec4 out_color;

void main()
{
    out_color = vec4(vec3(0.2), 1.0);
}
)";


const char vertex_shader_rect[] =
R"(#version 330 core

const vec2 vertices[4] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0,  1.0)
);

out vec3 position;

uniform mat4 view_projection_inverse;

void main()
{
    vec2 vertex = vertices[gl_VertexID];
    gl_Position = vec4(vertex, 0.0, 1.0);
    vec4 clip_space = view_projection_inverse * gl_Position;
    position = clip_space.xyz / clip_space.w;
}
)";

const char fragment_shader_rect[] =
R"(#version 330 core

in vec3 position;

uniform vec3 camera_position;
uniform sampler2D environment_texture;

const float PI = 3.141592653589793;

layout (location = 0) out vec4 out_color;

void main()
{
    vec3 dir = normalize(position - camera_position);

    float x = atan(dir.z, dir.x) / PI * 0.5 + 0.5;
    float y = -atan(dir.y, length(dir.xz)) / PI + 0.5;

    out_color = vec4(texture(environment_texture, vec2(x, y)).xyz, 1.0);
}
)";


GLuint create_shader(GLenum type, const char* source)
{
    GLuint result = glCreateShader(type);
    glShaderSource(result, 1, &source, nullptr);
    glCompileShader(result);
    GLint status;
    glGetShaderiv(result, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetShaderiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetShaderInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Shader compilation failed: " + info_log);
    }
    return result;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader)
{
    GLuint result = glCreateProgram();
    glAttachShader(result, vertex_shader);
    glAttachShader(result, fragment_shader);
    glLinkProgram(result);

    GLint status;
    glGetProgramiv(result, GL_LINK_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Program linkage failed: " + info_log);
    }

    return result;
}

struct vertex
{
    glm::vec3 position;
    glm::vec3 tangent;
    glm::vec3 normal;
    glm::vec2 texcoords;
};

std::pair<std::vector<vertex>, std::vector<std::uint32_t>> generate_sphere(float radius, int quality)
{
    std::vector<vertex> vertices;

    for (int latitude = -quality; latitude <= quality; ++latitude)
    {
        for (int longitude = 0; longitude <= 4 * quality; ++longitude)
        {
            float lat = (latitude * glm::pi<float>()) / (2.f * quality);
            float lon = (longitude * glm::pi<float>()) / (2.f * quality);

            auto& vertex = vertices.emplace_back();
            vertex.normal = { std::cos(lat) * std::cos(lon), std::sin(lat), std::cos(lat) * std::sin(lon) };
            vertex.position = vertex.normal * radius;
            vertex.tangent = { -std::cos(lat) * std::sin(lon), 0.f, std::cos(lat) * std::cos(lon) };
            vertex.texcoords.x = (longitude * 1.f) / (4.f * quality);
            vertex.texcoords.y = (latitude * 1.f) / (2.f * quality) + 0.5f;
        }
    }

    std::vector<std::uint32_t> indices;

    for (int latitude = 0; latitude < 2 * quality; ++latitude)
    {
        for (int longitude = 0; longitude < 4 * quality; ++longitude)
        {
            std::uint32_t i0 = (latitude + 0) * (4 * quality + 1) + (longitude + 0);
            std::uint32_t i1 = (latitude + 1) * (4 * quality + 1) + (longitude + 0);
            std::uint32_t i2 = (latitude + 0) * (4 * quality + 1) + (longitude + 1);
            std::uint32_t i3 = (latitude + 1) * (4 * quality + 1) + (longitude + 1);

            indices.insert(indices.end(), { i0, i1, i2, i2, i1, i3 });
        }
    }

    return { std::move(vertices), std::move(indices) };
}

GLuint load_texture(std::string const& path)
{
    int width, height, channels;
    auto pixels = stbi_load(path.data(), &width, &height, &channels, 4);

    GLuint result;
    glGenTextures(1, &result);
    glBindTexture(GL_TEXTURE_2D, result);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(pixels);

    return result;
}

int main() try
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window* window = SDL_CreateWindow("Graphics course practice 5",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        800, 600,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    glClearColor(0.8f, 0.8f, 1.f, 0.f);


    const float W = 7873, D = 5280;
    const int GRID = 1024;

    // Water
    std::vector<uint32_t> indices;
    indices.reserve((GRID - 1)* (GRID - 1) * 6);

    for (int z = 0; z < GRID - 1; ++z)
    {
        for (int x = 0; x < GRID - 1; ++x)
        {
            uint32_t i0 = z * GRID + x;
            uint32_t i1 = i0 + 1;
            uint32_t i2 = i0 + GRID;
            uint32_t i3 = i2 + 1;

            indices.push_back(i0);
            indices.push_back(i2);
            indices.push_back(i1);

            indices.push_back(i1);
            indices.push_back(i2);
            indices.push_back(i3);
        }
    }


    auto water_vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_water);
    auto water_fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_water);
    auto water_program = create_program(water_vertex_shader, water_fragment_shader);

    GLuint water_model_location = glGetUniformLocation(water_program, "model");
    GLuint water_view_location = glGetUniformLocation(water_program, "view");
    GLuint water_projection_location = glGetUniformLocation(water_program, "projection");
    GLuint water_time_location = glGetUniformLocation(water_program, "time");
    GLuint bottom_world_box_location = glGetUniformLocation(water_program, "bottom_world_box");
    GLuint water_light_direction_location = glGetUniformLocation(water_program, "light_direction");
    GLuint water_camera_position_location = glGetUniformLocation(water_program, "camera_position");
    GLuint water_bottom_texture_location = glGetUniformLocation(water_program, "bottom_texture");
    GLuint water_caustics_texture_location = glGetUniformLocation(water_program, "caustics_texture");
    GLuint water_environment_texture_location = glGetUniformLocation(water_program, "environment_texture");


    GLuint water_vao, water_ebo;
    glGenVertexArrays(1, &water_vao);
    glGenBuffers(1, &water_ebo);

    glBindVertexArray(water_vao);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, water_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t), indices.data(), GL_STATIC_DRAW);

    //

    // Bottom
    
    auto bottom_vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_bottom);
    auto bottom_fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_bottom);
    auto bottom_program = create_program(bottom_vertex_shader, bottom_fragment_shader);

    GLuint bottom_texture_location = glGetUniformLocation(bottom_program, "bottom_texture");
    GLuint bottom_caustics_texture_location = glGetUniformLocation(bottom_program, "caustics_texture");
    GLuint bottom_model_location = glGetUniformLocation(bottom_program, "model");
    GLuint bottom_view_location = glGetUniformLocation(bottom_program, "view");
    GLuint bottom_projection_location = glGetUniformLocation(bottom_program, "projection");
    GLuint bottom_light_direction_location = glGetUniformLocation(bottom_program, "light_direction");
    GLuint bottom_camera_position_location = glGetUniformLocation(bottom_program, "camera_position");

    GLuint bottom_vao;
    glGenVertexArrays(1, &bottom_vao);
    //

    // Caustics

    auto caustics_vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_caustics);
    auto caustics_fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_caustics);
    auto caustics_program = create_program(caustics_vertex_shader, caustics_fragment_shader);

    GLuint caustics_time_location = glGetUniformLocation(caustics_program, "time");
    GLuint caustics_light_direction_location = glGetUniformLocation(caustics_program, "light_direction");

    const int CAUSTICS_RES = 1024;

    GLuint caustics_texture;
    glGenTextures(1, &caustics_texture);
    glBindTexture(GL_TEXTURE_2D, caustics_texture);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_R16F,
        CAUSTICS_RES,
        CAUSTICS_RES,
        0,
        GL_RED,
        GL_FLOAT,
        nullptr
    );

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);


    GLuint caustics_fbo;
    glGenFramebuffers(1, &caustics_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, caustics_fbo);

    glFramebufferTexture2D(
        GL_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D,
        caustics_texture,
        0
    );
    //

    // Environment map
    auto rect_vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_rect);
    auto rect_fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_rect);
    auto rect_program = create_program(rect_vertex_shader, rect_fragment_shader);

    GLuint camera_position_location_rect = glGetUniformLocation(rect_program, "camera_position");
    GLuint environment_texture_location_rect = glGetUniformLocation(rect_program, "environment_texture");
    GLuint view_projection_inverse = glGetUniformLocation(rect_program, "view_projection_inverse");

    GLuint rect_vao;
    glGenVertexArrays(1, &rect_vao);
    //


    GLuint sphere_vao, sphere_vbo, sphere_ebo;
    glGenVertexArrays(1, &sphere_vao);
    glBindVertexArray(sphere_vao);
    glGenBuffers(1, &sphere_vbo);
    glGenBuffers(1, &sphere_ebo);
    GLuint sphere_index_count;
    {
        auto [vertices, indices] = generate_sphere(1.f, 16);

        glBindBuffer(GL_ARRAY_BUFFER, sphere_vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]), indices.data(), GL_STATIC_DRAW);

        sphere_index_count = indices.size();
    }
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, position));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, tangent));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, normal));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void*)offsetof(vertex, texcoords));

    std::string project_root = PROJECT_ROOT;
    GLuint environment_texture = load_texture(project_root + "/textures/environment_map2.jpg");
    GLuint bottom_texture = load_texture(project_root + "/textures/bottom.jpg");

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;

    float view_elevation = glm::radians(30.f);
    float view_azimuth = 0.f;
    float camera_distance = 2.f;

    const glm::vec3 vertices[4] = {
        glm::vec3(-1, 0, -D / W),
        glm::vec3(1, 0, -D / W),
        glm::vec3(-1, 0, D / W),
        glm::vec3(1, 0, D / W)
    };



    bool running = true;
    while (running)
    {
        for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type)
        {
        case SDL_QUIT:
            running = false;
            break;
        case SDL_WINDOWEVENT: switch (event.window.event)
        {
        case SDL_WINDOWEVENT_RESIZED:
            width = event.window.data1;
            height = event.window.data2;
            glViewport(0, 0, width, height);
            break;
        }
                            break;
        case SDL_KEYDOWN:
            button_down[event.key.keysym.sym] = true;
            break;
        case SDL_KEYUP:
            button_down[event.key.keysym.sym] = false;
            break;
        }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;
        time += dt;

        if (button_down[SDLK_UP])
            camera_distance -= 4.f * dt;
        if (button_down[SDLK_DOWN])
            camera_distance += 4.f * dt;

        if (button_down[SDLK_LEFT])
            view_azimuth -= 2.f * dt;
        if (button_down[SDLK_RIGHT])
            view_azimuth += 2.f * dt;

        if (button_down[SDLK_w])
            view_elevation += 2.f * dt;
        if (button_down[SDLK_s])
            view_elevation -= 2.f * dt;



        float near = 0.1f;
        float far = 100.f;
        float top = near;
        float right = (top * width) / height;

        glm::mat4 model = glm::mat4(1.f);

        glm::mat4 view(1.f);
        view = glm::translate(view, { 0.f, 0.f, -camera_distance });
        view = glm::rotate(view, view_elevation, { 1.f, 0.f, 0.f });
        view = glm::rotate(view, view_azimuth, { 0.f, 1.f, 0.f });

        glm::mat4 projection = glm::mat4(1.f);
        projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

        glm::vec3 light_direction = glm::normalize(-glm::vec3(2.f, 2.f, -1.f));

        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        glm::vec3 bottom_world_box[4];

        for (int i = 0; i < 4; i++) {
            glm::vec3 pos = (model * glm::vec4(vertices[i], 1.0)).xyz;
            bottom_world_box[i] = projection * view * glm::vec4(pos, 1.0);
        }

        // Caustics
        glUseProgram(caustics_program);
        glUniform1f(caustics_time_location, time);
        glUniform3fv(caustics_light_direction_location, 1, reinterpret_cast<float*>(&light_direction));

        glBindVertexArray(water_vao);

        glBindFramebuffer(GL_FRAMEBUFFER, caustics_fbo);
        glViewport(0, 0, CAUSTICS_RES, CAUSTICS_RES);

        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);

        glDisable(GL_DEPTH_TEST);

        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);

        glClearColor(0.8f, 0.8f, 1.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //

        // Environment map

        glDisable(GL_DEPTH_TEST);

        glUseProgram(rect_program);
        glUniform1i(environment_texture_location_rect, 0);
        glUniform3fv(camera_position_location_rect, 1, reinterpret_cast<float*>(&camera_position));
        glUniformMatrix4fv(view_projection_inverse, 1, GL_FALSE, reinterpret_cast<float*>(&(glm::inverse(projection* view)[0][0])));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, environment_texture);

        glBindVertexArray(rect_vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glEnable(GL_DEPTH_TEST);
        //

        // Bottom
        glUseProgram(bottom_program);
        glUniform1i(bottom_texture_location, 0);
        glUniform1i(bottom_caustics_texture_location, 1);
        glUniformMatrix4fv(bottom_model_location, 1, GL_FALSE, reinterpret_cast<float*>(&model));
        glUniformMatrix4fv(bottom_view_location, 1, GL_FALSE, reinterpret_cast<float*>(&view));
        glUniformMatrix4fv(bottom_projection_location, 1, GL_FALSE, reinterpret_cast<float*>(&projection));
        glUniform3fv(bottom_light_direction_location, 1, reinterpret_cast<float*>(&light_direction));
        glUniform3fv(bottom_camera_position_location, 1, reinterpret_cast<float*>(&camera_position));


        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, bottom_texture);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, caustics_texture);

        glBindVertexArray(bottom_vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        //


        // Water
        glUseProgram(water_program);
        glUniform1i(water_bottom_texture_location, 0);
        glUniform1i(water_caustics_texture_location, 1);
        glUniform1i(water_environment_texture_location, 2);
        glUniform1f(water_time_location, time);
        glUniformMatrix4fv(water_model_location, 1, GL_FALSE, reinterpret_cast<float*>(&model));
        glUniformMatrix4fv(water_view_location, 1, GL_FALSE, reinterpret_cast<float*>(&view));
        glUniformMatrix4fv(water_projection_location, 1, GL_FALSE, reinterpret_cast<float*>(&projection));
        glUniform3fv(bottom_world_box_location, 4, &bottom_world_box[0].x);
        glUniform3fv(water_light_direction_location, 1, reinterpret_cast<float*>(&light_direction));
        glUniform3fv(water_camera_position_location, 1, reinterpret_cast<float*>(&camera_position));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, bottom_texture);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, caustics_texture);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, environment_texture);

        glBindVertexArray(water_vao);

        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        //

        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
}
catch (std::exception const& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
