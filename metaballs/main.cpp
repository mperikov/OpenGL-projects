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
#include <random>
#include <map>
#include <cmath>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/quaternion.hpp>
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
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}

const char vertex_shader_source[] =
R"(#version 330 core

void main() {
}
)";

const char geometry_shader_source[] =
R"(#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 24) out;

uniform float K;
uniform int resolution;

uniform mat4 MVP;

uniform sampler3D rgba_texture;

out vec3 color;
out vec3 position;
out vec3 normal;

vec3 texToWorld(vec3 t) {
    return mix(vec3(-1.0), vec3(1.0), t);
}

float sampleField(vec3 t) {
    return texture(rgba_texture, t).a;
}

vec3 gradient(vec3 t) {
    float h = 1.0 / float(resolution);
    float dx = sampleField(t + vec3(h,0,0)) - sampleField(t - vec3(h,0,0));
    float dy = sampleField(t + vec3(0,h,0)) - sampleField(t - vec3(0,h,0));
    float dz = sampleField(t + vec3(0,0,h)) - sampleField(t - vec3(0,0,h));
    return vec3(dx, dy, dz);
}

vec3 interp(vec3 p1, vec3 p2, float v1, float v2) {
    float t = (K - v1) / (v2 - v1);
    return mix(p1, p2, t);
}

void emit1(vec3 t) {
    position = texToWorld(t);
    normal = -normalize(gradient(t));
    gl_Position = MVP * vec4(position, 1.0);
    vec4 tex = texture(rgba_texture, t);
    color = tex.rgb / max(tex.a, 1e-6);
    EmitVertex();
}

void emit3(vec3 t0, vec3 t1, vec3 t2) {
    emit1(t0);
    emit1(t1);
    emit1(t2);
    EndPrimitive();
}

void emit4(vec3 t0, vec3 t1, vec3 t2, vec3 t3) {
    emit1(t0);
    emit1(t1);
    emit1(t2);
    emit1(t3);
    EndPrimitive();
}

void main() {
    int id = gl_PrimitiveIDIn;

    int cells = resolution;
    int z = id / (cells * cells);
    int y = (id / cells) % cells;
    int x = id % cells;

    ivec3 cell = ivec3(x, y, z);

    vec3 p[8];
    float v[8];

    for (int i = 0; i < 8; i++) {
        ivec3 o = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        vec3 t = (vec3(cell + o)) / float(resolution - 1);
        p[i] = t;
        v[i] = sampleField(t);
    }


    const int tetra[6][4] = int[6][4](
        int[4](0,1,2,6),
        int[4](0,1,4,6),
        int[4](2,1,3,6),
        int[4](1,7,3,6),
        int[4](1,5,7,6),
        int[4](1,5,4,6)
    );

    for (int t = 0; t < 6; t++) {
        int ids[4] = tetra[t];

        int inside[4];
        int count = 0;
        for (int i = 0; i < 4; i++) {
            inside[i] = v[ids[i]] > K ? 1 : 0;
            count += inside[i];
        }

        if (count == 0 || count == 4)
            continue;

        vec3 pts[6];
        int pc = 0;

        for (int i = 0; i < 4; i++)
        for (int j = i+1; j < 4; j++) {
            if (inside[i] != inside[j]) {
                pts[pc++] = interp(
                    p[ids[i]], p[ids[j]],
                    v[ids[i]], v[ids[j]]
                );
            }
        }

        if (pc == 3) {
            emit3(
                pts[0],
                pts[1],
                pts[2]
            );
        } else if (pc == 4) {
            emit4(
                pts[0],
                pts[1],
                pts[2],
                pts[3]
            );
        }
    }
}
)";

const char fragment_shader_source[] =
R"(#version 330 core

in vec3 color;
in vec3 position;
in vec3 normal;

out vec4 out_color;

uniform vec3 light_position;
uniform vec3 camera_position;

const vec3 ambient_light_color = vec3(0.3);
const vec3 light_color = vec3(1.0);
const float specular_power = 64;
const vec3 specular_light_color = vec3(0.6);

void main() {
    vec3 norm = normalize(normal);
    vec3 light_direction = normalize(light_position - position);
    vec3 view_direction = normalize(camera_position - position);
    
    vec3 albedo = color;

    float cosine = dot(norm, light_direction);
    float light_factor = max(0.0, cosine);
    vec3 reflected_direction = 2.0 * norm * cosine - light_direction;


    vec3 ambient = ambient_light_color * albedo;
    vec3 diffuse = light_color * albedo * light_factor;
    vec3 specular = specular_light_color * pow(max(dot(reflected_direction, view_direction), 0.0), specular_power);

    out_color = vec4(ambient + diffuse + specular, 1.0);
}


)";



const char compute_shader_source[] =
R"(#version 430 core

layout (local_size_x = 4,
        local_size_y = 4,
        local_size_z = 4) in;

struct Metaball {
    vec3 position;
    float radius;
    vec3 color;
    float pad;
};

layout(std430, binding = 0) buffer Metaballs {
    Metaball balls[];
};

layout(rgba32f, binding = 0) uniform image3D rgba_texture;

uniform int balls_count;
uniform int resolution;

void main() {
    ivec3 id = ivec3(gl_GlobalInvocationID);

    if (id.x >= resolution || id.y >= resolution || id.z >= resolution)
        return;

    vec3 p = mix(vec3(-1.0), vec3(1.0), vec3(id) / float(resolution - 1));

    float sum_w = 0.0;
    vec3 sum_c = vec3(0.0);

    for (int i = 0; i < balls_count; ++i) {
        vec3 d = p - balls[i].position;
        float r = balls[i].radius;
        float w = exp(-length(d) / r);
        sum_w += w;
        sum_c += w * balls[i].color;
    }

    imageStore(rgba_texture, id, vec4(sum_c, sum_w));
}

)";


GLuint create_shader(GLenum type, const char * source)
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

template <typename ... Shaders>
GLuint create_program(Shaders ... shaders)
{
    GLuint result = glCreateProgram();
    (glAttachShader(result, shaders), ...);
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

struct particle
{
    glm::vec3 position;
    float size;
    float angle;
    glm::vec3 velocity;
    float angular_velocity;
};

particle create_particle(std::default_random_engine& rng) {
    particle p;

    p.position.x = std::uniform_real_distribution<float>{ -0.25, 0.25 }(rng);
    p.position.y = 0.f;
    p.position.z = std::uniform_real_distribution<float>{ -1.f, 1.f }(rng);

    p.size = std::uniform_real_distribution<float>{ 0.2, 0.3 }(rng);
    p.velocity = glm::vec3(std::uniform_real_distribution<float>{ -0.1, 0.1 }(rng));
    p.velocity.y = std::max(0.f, std::normal_distribution<float>{0, 1 }(rng));

    p.angle = std::uniform_real_distribution<float>{ 0, 3.14 }(rng);
    p.angular_velocity = std::uniform_real_distribution<float>{ -1, 1 }(rng);

    return p;
}

struct Metaball {
    glm::vec3 position;
    float radius;
    glm::vec3 color;
    float pad;
};

std::vector<Metaball> balls;
std::vector<glm::vec3> dir;

glm::vec3 rand_vec3(std::default_random_engine& rng) {
    glm::vec3 ret;
    ret.x = std::uniform_real_distribution<float>{ 0.0, 1.0 }(rng);
    ret.y = std::uniform_real_distribution<float>{ 0.0, 1.0 }(rng);
    ret.z = std::uniform_real_distribution<float>{ 0.0, 1.0 }(rng);
    return ret;
}



void add_metaball(std::default_random_engine& rng) {
    
    Metaball ret;

    ret.position = rand_vec3(rng) * 2.f - 1.f;
    ret.position.z /= 2.f;

    ret.radius = std::uniform_real_distribution<float>{ 0.1, 0.2 }(rng);
    
    ret.color = rand_vec3(rng);

    balls.push_back(ret);

    dir.push_back(glm::normalize(rand_vec3(rng) * 2.f - 1.f));
}

void del_metaball() {
    if (!balls.empty()) {
        balls.pop_back();
        dir.pop_back();
    }
}

void replace_meataball(int i, std::default_random_engine& rng) {
    Metaball ret;

    ret.position = rand_vec3(rng) * 2.f - 1.f;
    ret.position.z /= 2.f;

    ret.radius = std::uniform_real_distribution<float>{ 0.03, 0.11 }(rng);

    ret.color = rand_vec3(rng);

    balls[i] = ret;

    dir[i] = glm::normalize(rand_vec3(rng) * 2.f - 1.f);
}


int main() try
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 11",
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

    const int resolution = 128;


    std::default_random_engine rng;

    for (int i = 0; i < 10;i++)
        add_metaball(rng);
    
    GLuint ssbo;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        balls.size() * sizeof(Metaball),
        balls.data(),
        GL_DYNAMIC_DRAW
    );
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_3D, texture);

    glTexStorage3D(
        GL_TEXTURE_3D,
        1,
        GL_RGBA32F,
        resolution,
        resolution,
        resolution
    );

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);



    auto compute_shader = create_shader(GL_COMPUTE_SHADER, compute_shader_source);
    auto compute_program = create_program(compute_shader);

    GLuint balls_count_location = glGetUniformLocation(compute_program, "balls_count");
    GLuint resolution_location = glGetUniformLocation(compute_program, "resolution");

    

    
    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto geometry_shader = create_shader(GL_GEOMETRY_SHADER, geometry_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, geometry_shader, fragment_shader);

    GLuint MVP_location = glGetUniformLocation(program, "MVP");
    GLuint geometry_resolution_location = glGetUniformLocation(program, "resolution");
    GLuint texture_location = glGetUniformLocation(program, "rgba_texture");
    GLuint isolevel_location = glGetUniformLocation(program, "K");
    GLuint light_position_location = glGetUniformLocation(program, "light_position");
    GLuint camera_position_location = glGetUniformLocation(program, "camera_position");

    GLuint vao;
    glGenVertexArrays(1, &vao);


    glPointSize(5.f);

    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;

    std::map<SDL_Keycode, bool> button_down;

    float view_angle = 0.f;
    float camera_distance = 2.f;
    float camera_height = 0.5f;

    float camera_rotation = 0.f;

    bool paused = false;

    float K = 0.2;
    int GRID_SIZE = 16;

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
            if (event.key.keysym.sym == SDLK_SPACE)
                paused = !paused;
            if (event.key.keysym.sym == SDLK_a)
                GRID_SIZE = std::min(128, GRID_SIZE * 2);
            if (event.key.keysym.sym == SDLK_d)
                GRID_SIZE = std::max(2, GRID_SIZE / 2);
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
            camera_distance -= 3.f * dt;
        if (button_down[SDLK_DOWN])
            camera_distance += 3.f * dt;

        if (button_down[SDLK_LEFT])
            camera_rotation -= 3.f * dt;
        if (button_down[SDLK_RIGHT])
            camera_rotation += 3.f * dt;

        if (button_down[SDLK_w])
            view_angle += 2.f * dt;
        if (button_down[SDLK_s])
            view_angle -= 2.f * dt;

        if (button_down[SDLK_k])
            K += dt * 0.2;
        if (button_down[SDLK_l])
            K -= dt * 0.2;



        for (int i = 0; i < balls.size(); i++) {
            balls[i].position += dt * 0.01f / balls[i].radius * dir[i];
            
            bool fl = 0;
            for (int j = 0; j < 3; j++) {
                if (abs(balls[i].position[j]) > 1 + balls[i].radius)
                    fl = 1;
            }
            if (fl)
                replace_meataball(i, rng);
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        float near = 0.1f;
        float far = 100.f;

        glm::mat4 model(1.f);

        glm::mat4 view(1.f);
        view = glm::translate(view, {0.f, -camera_height, -camera_distance});
        view = glm::rotate(view, view_angle, {1.f, 0.f, 0.f});
        view = glm::rotate(view, camera_rotation, {0.f, 1.f, 0.f});

        glm::mat4 projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();
        glm::vec3 light_position = glm::vec3(sin(0.5 * time) * 2.0, 2.0, cos(0.5 * time) * 2.0);
        //glm::vec3 light_position = glm::vec3(2.0, 2.0, 0.0);

        
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glBufferSubData(
            GL_SHADER_STORAGE_BUFFER,
            0,
            balls.size() * sizeof(Metaball),
            balls.data()
        );

        glUseProgram(compute_program);

        glUniform1i(balls_count_location, balls.size());
        glUniform1i(resolution_location, resolution);

        glBindImageTexture(
            0,
            texture,
            0,
            GL_TRUE,
            0,
            GL_WRITE_ONLY,
            GL_RGBA32F
        );

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

        glDispatchCompute(
            resolution / 4,
            resolution / 4,
            resolution / 4
        );

        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        


        glUseProgram(program);

        glm::mat4 MVP = projection * view * model;

        
        glUniformMatrix4fv(MVP_location, 1, GL_FALSE, reinterpret_cast<float*>(&MVP));
        glUniform1f(isolevel_location, K);
        glUniform1i(texture_location, 0);
        glUniform1i(geometry_resolution_location, GRID_SIZE);
        glUniform3fv(camera_position_location, 1, reinterpret_cast<float*>(&camera_position));
        glUniform3fv(light_position_location, 1, reinterpret_cast<float*>(&light_position));


        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, texture);

        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, GRID_SIZE * GRID_SIZE * GRID_SIZE);

        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
}
catch (std::exception const & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
