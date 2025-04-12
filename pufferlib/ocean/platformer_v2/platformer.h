#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "raylib.h"

#define MAX_STEPS 1000
const int GOAL_REWARD = 5.0;
const unsigned char DONE = 1;
const unsigned char NOT_DONE = 0;

const float GRAVITY = 0.5f;
const float JUMP_FORCE = -10.0f;
const float MOVE_SPEED = 5.0f;

// Actions
const int ACTION_NOOP = 0;
const int ACTION_LEFT = 1;
const int ACTION_RIGHT = 2;
const int ACTION_JUMP = 3;

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float success_rate;
};

#define LOG_BUFFER_SIZE 1024
typedef struct LogBuffer LogBuffer;
struct LogBuffer {
    Log* logs;
    int length;
    int idx;
};

LogBuffer* allocate_logbuffer(int size) {
    LogBuffer* logs = (LogBuffer*)calloc(1, sizeof(LogBuffer));
    logs->logs = (Log*)calloc(size, sizeof(Log));
    logs->length = size;
    logs->idx = 0;
    return logs;
}

void free_logbuffer(LogBuffer* buffer) {
    free(buffer->logs);
    free(buffer);
}

void add_log(LogBuffer* logs, Log* log) {
    if (logs->idx == logs->length) {
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return += logs->logs[i].episode_return;
        log.episode_length += logs->logs[i].episode_length;
        log.success_rate += logs->logs[i].success_rate;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.success_rate /= logs->idx;
    logs->idx = 0;
    return log;
}

typedef struct Player Player;
struct Player {
    float x;
    float y;
    float vel_x;
    float vel_y;
    int width;
    int height;
    int on_ground;
};

typedef struct Goal Goal;
struct Goal {
    float x;
    float y;
    int width;
    int height;
};

typedef struct Enemy Enemy;
struct Enemy {
    float x;
    float y;
    int width;
    int height;
};

typedef struct Platform Platform;
struct Platform {
    float x;
    float y;
    int width;
    int height;
};

typedef struct CPlatformer CPlatformer;
struct CPlatformer {
    // Pufferlib inputs/outputs
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;
    Log log;
    
    // Game state
    Player player;
    Goal goal;
    Enemy enemy;
    Platform ground;
    
    // Game metrics
    int steps;
    float episode_return;
    float previous_distance_to_goal;
    
    // Rendering configuration
    int screen_width;
    int screen_height;
};

void allocate_cplatformer(CPlatformer* env) {
    env->observations = (float*)calloc(6, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->dones = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_cplatformer(CPlatformer* env) {
    free_logbuffer(env->log_buffer);
}

void free_allocated_cplatformer(CPlatformer* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free_cplatformer(env);
}

int check_collision(float x1, float y1, int w1, int h1, float x2, float y2, int w2, int h2) {
    return (x1 < x2 + w2 && x1 + w1 > x2 && y1 < y2 + h2 && y1 + h1 > y2);
}

void compute_observation(CPlatformer* env) {
    env->observations[0] = env->player.x / env->screen_width;
    env->observations[1] = env->player.y / env->screen_height;
    env->observations[2] = env->player.vel_x / 20.0f; // Normalize velocity
    env->observations[3] = env->player.vel_y / 20.0f;
    env->observations[4] = env->enemy.x / env->screen_width;
    env->observations[5] = env->enemy.y / env->screen_height;
}

// Helper function to calculate distance between two points
float calculate_distance(float x1, float y1, float x2, float y2) {
    return sqrtf(powf(x2 - x1, 2) + powf(y2 - y1, 2));
}

void c_reset(CPlatformer* env) {
    // Initialize log
    env->log = (Log){0};
    env->dones[0] = NOT_DONE;
    
    // Reset player
    env->player.x = 50;
    env->player.y = env->screen_height - env->player.height - env->ground.height;
    env->player.vel_x = 0;
    env->player.vel_y = 0;
    env->player.on_ground = 1;
    
    // Set up goal position
    env->goal.x = env->screen_width - 100;
    env->goal.y = env->screen_height - env->goal.height - env->ground.height;
    
    // Set up enemy position
    env->enemy.x = env->screen_width / 2; // Place enemy in the middle
    env->enemy.y = env->screen_height - env->enemy.height - env->ground.height;
    
    // Calculate initial distance to goal
    env->previous_distance_to_goal = calculate_distance(
        env->player.x + env->player.width/2, 
        env->player.y + env->player.height/2,
        env->goal.x + env->goal.width/2, 
        env->goal.y + env->goal.height/2
    );
    
    // Reset metrics
    env->steps = 0;
    env->episode_return = 0;
    
    // Reset reward
    env->rewards[0] = 0;
    
    // Initialize ground
    env->ground.x = 0;
    env->ground.y = env->screen_height - env->ground.height;
    env->ground.width = env->screen_width;
    
    // Set initial observation
    compute_observation(env);
}

void finish_game(CPlatformer* env, float reward) {
    env->rewards[0] = reward;
    env->dones[0] = DONE;
    env->episode_return += reward;
    
    // Log episode stats
    env->log.episode_return = env->episode_return;
    env->log.episode_length = env->steps;
    env->log.success_rate = (reward > 0) ? 1.0 : 0.0;
    add_log(env->log_buffer, &env->log);
}

void c_step(CPlatformer* env) {
    // Update episode length
    env->log.episode_length += 1;
    
    // Default no reward
    env->rewards[0] = 0;
    
    // Check if already done
    if (env->dones[0] == DONE) {
        add_log(env->log_buffer, &env->log);
        c_reset(env);
        return;
    }
    
    // Process the action
    int action = env->actions[0];
    
    // Update player velocity based on action
    if (action == ACTION_LEFT) {
        env->player.vel_x = -MOVE_SPEED;
    } else if (action == ACTION_RIGHT) {
        env->player.vel_x = MOVE_SPEED;
    } else {
        // Decelerate if no movement action
        env->player.vel_x *= 0.8f;
    }
    
    // Jump if on ground
    if (action == ACTION_JUMP && env->player.on_ground) {
        env->player.vel_y = JUMP_FORCE;
        env->player.on_ground = 0;
    }
    
    // Apply gravity
    env->player.vel_y += GRAVITY;
    
    // Update position
    env->player.x += env->player.vel_x;
    env->player.y += env->player.vel_y;
    
    // Handle ground collision
    if (env->player.y + env->player.height > env->ground.y) {
        env->player.y = env->ground.y - env->player.height;
        env->player.vel_y = 0;
        env->player.on_ground = 1;
    }
    
    // Keep player in bounds
    if (env->player.x < 0) {
        env->player.x = 0;
        env->player.vel_x = 0;
    }
    if (env->player.x + env->player.width > env->screen_width) {
        env->player.x = env->screen_width - env->player.width;
        env->player.vel_x = 0;
    }
    
    // Calculate current distance to goal
    float current_distance = calculate_distance(
        env->player.x + env->player.width/2, 
        env->player.y + env->player.height/2,
        env->goal.x + env->goal.width/2, 
        env->goal.y + env->goal.height/2
    );
    
    // Calculate proximity reward
    if (current_distance < env->previous_distance_to_goal) {
        env->rewards[0] = 1.0; // Reward for getting closer
    } else if (current_distance > env->previous_distance_to_goal) {
        env->rewards[0] = -1.0; // Penalty for getting farther
    }
    
    // Update previous distance
    env->previous_distance_to_goal = current_distance;
    
    // Update observation
    compute_observation(env);
    
    // Check for goal collision
    if (check_collision(env->player.x, env->player.y, env->player.width, env->player.height,
                       env->goal.x, env->goal.y, env->goal.width, env->goal.height)) {
        finish_game(env, GOAL_REWARD);
        return;
    }
    
    // Check for enemy collision
    if (check_collision(env->player.x, env->player.y, env->player.width, env->player.height,
                      env->enemy.x, env->enemy.y, env->enemy.width, env->enemy.height)) {
        finish_game(env, -5.0); // -5 reward for hitting enemy
        return;
    }
    
    // Update episode return with current reward
    env->episode_return += env->rewards[0];
    
    // Check if max steps reached
    env->steps++;
    if (env->steps >= MAX_STEPS) {
        finish_game(env, 0);
        return;
    }
}

typedef struct Client Client;
struct Client {
    int width;
    int height;
};

Client* make_client(int width, int height) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = width;
    client->height = height;
    
    InitWindow(width, height, "Platformer");
    SetTargetFPS(60);
    
    return client;
}

void c_render(Client* client, CPlatformer* env) {
    BeginDrawing();
    ClearBackground(RAYWHITE);
    
    // Draw ground
    DrawRectangle(env->ground.x, env->ground.y, env->ground.width, env->ground.height, DARKGRAY);
    
    // Draw goal
    DrawRectangle(env->goal.x, env->goal.y, env->goal.width, env->goal.height, GREEN);
    
    // Draw enemy
    DrawRectangle(env->enemy.x, env->enemy.y, env->enemy.width, env->enemy.height, RED);
    
    // Draw player
    DrawRectangle(env->player.x, env->player.y, env->player.width, env->player.height, BLUE);
    
    // Draw stats
    char stats[100];
    sprintf(stats, "Steps: %d", env->steps);
    DrawText(stats, 10, 10, 20, BLACK);
    
    EndDrawing();
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
} 