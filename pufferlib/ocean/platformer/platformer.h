#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "raylib.h"

// Platformer constants
#define GRAVITY 0.5f
#define JUMP_FORCE 13.0f
#define PLAYER_SPEED 5.0f
#define PLATFORM_COUNT 5
#define COIN_COUNT 4  // Number of coins (excluding ground)
#define GROUND_HEIGHT 40.0f
#define COIN_RADIUS 10.0f

// Observation size (player x,y + velocity + platforms x,y + coins x,y)
#define OBS_SIZE (2 + 2 + PLATFORM_COUNT * 2 + COIN_COUNT * 2)

const unsigned char DONE = 1;
const unsigned char NOT_DONE = 0;

// Game state
typedef enum { PLAYING = 0, WON = 1, LOST = 2 } GameState;

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float score;
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
        log.score += logs->logs[i].score;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    logs->idx = 0;
    return log;
}

// Platform structure
typedef struct Platform {
    float x;
    float y;
    float width;
    float height;
} Platform;

// Coin structure
typedef struct Coin {
    float x;
    float y;
    float radius;
    bool collected;
} Coin;

// Action type (4 possible actions: 0=left, 1=right, 2=jump, 3=no action)
typedef enum { LEFT = 0, RIGHT = 1, JUMP = 2, NOOP = 3, ACTIONS_COUNT = 4 } Action;

typedef struct CPlatformer CPlatformer;
struct CPlatformer {
    // Pufferlib inputs / outputs
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;
    Log log;

    // Player state
    float player_x;
    float player_y;
    float player_vx;
    float player_vy;
    float player_width;
    float player_height;
    bool on_ground;
    
    // Environment state
    Platform platforms[PLATFORM_COUNT];
    Coin coins[COIN_COUNT];
    int tick;
    int max_ticks;
    GameState state;
    
    // Rendering configuration
    int width;
    int height;
};

void allocate_cplatformer(CPlatformer* env) {
    env->observations = (float*)calloc(OBS_SIZE, sizeof(float));
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

// Check collision between player and platform
bool check_collision(float player_x, float player_y, float player_width, float player_height,
                    float plat_x, float plat_y, float plat_width, float plat_height) {
    return (player_x < plat_x + plat_width &&
            player_x + player_width > plat_x &&
            player_y < plat_y + plat_height &&
            player_y + player_height > plat_y);
}

// Check if player is standing on a platform
bool check_standing(float player_x, float player_y, float player_width, float player_height,
                   float plat_x, float plat_y, float plat_width, float plat_height) {
    return (player_x + player_width > plat_x &&
            player_x < plat_x + plat_width &&
            player_y + player_height <= plat_y + 1 &&
            player_y + player_height >= plat_y - 1);
}

// Check coin collision with player
bool check_coin_collision(float player_x, float player_y, float player_width, float player_height,
                         float coin_x, float coin_y, float coin_radius) {
    // Check if player rectangle overlaps with coin circle
    // Find closest point on rectangle to circle center
    float closest_x = fmaxf(player_x, fminf(coin_x, player_x + player_width));
    float closest_y = fmaxf(player_y, fminf(coin_y, player_y + player_height));
    
    // Calculate distance between closest point and circle center
    float distance_x = coin_x - closest_x;
    float distance_y = coin_y - closest_y;
    float distance_squared = distance_x * distance_x + distance_y * distance_y;
    
    return distance_squared < (coin_radius * coin_radius);
}

// Compute observation vector
void compute_observation(CPlatformer* env) {
    // Normalize positions to [0,1]
    env->observations[0] = env->player_x / env->width;
    env->observations[1] = env->player_y / env->height;
    env->observations[2] = env->player_vx / 20.0f;  // Normalize velocity
    env->observations[3] = env->player_vy / 20.0f;  // Normalize velocity
    
    // Add platform positions
    for (int i = 0; i < PLATFORM_COUNT; i++) {
        env->observations[4 + i*2] = env->platforms[i].x / env->width;
        env->observations[4 + i*2 + 1] = env->platforms[i].y / env->height;
    }
    
    // Add coin positions (or -1,-1 if collected)
    for (int i = 0; i < COIN_COUNT; i++) {
        if (env->coins[i].collected) {
            env->observations[4 + PLATFORM_COUNT*2 + i*2] = -1.0f;
            env->observations[4 + PLATFORM_COUNT*2 + i*2 + 1] = -1.0f;
        } else {
            env->observations[4 + PLATFORM_COUNT*2 + i*2] = env->coins[i].x / env->width;
            env->observations[4 + PLATFORM_COUNT*2 + i*2 + 1] = env->coins[i].y / env->height;
        }
    }
}

// Initialize platforms with fixed positions
void init_platforms(CPlatformer* env) {
    // First platform is the ground
    env->platforms[0].x = 0;
    env->platforms[0].y = env->height - GROUND_HEIGHT;
    env->platforms[0].width = env->width;
    env->platforms[0].height = GROUND_HEIGHT;
    
    // Fixed platform layout - carefully designed to be reachable with JUMP_FORCE=13.0
    
    // Platform 1 - left side, low height
    env->platforms[1].width = 120;
    env->platforms[1].height = 20;
    env->platforms[1].x = 100;
    env->platforms[1].y = env->height - 200;
    
    // Platform 2 - right side, low height
    env->platforms[2].width = 120;
    env->platforms[2].height = 20;
    env->platforms[2].x = env->width - 220;
    env->platforms[2].y = env->height - 200;
    
    // Platform 3 - center, medium height
    env->platforms[3].width = 150;
    env->platforms[3].height = 20;
    env->platforms[3].x = (env->width - 150) / 2;
    env->platforms[3].y = env->height - 320;
    
    // Platform 4 - upper platform, reachable from center platform
    env->platforms[4].width = 100;
    env->platforms[4].height = 20;
    env->platforms[4].x = 150;
    env->platforms[4].y = env->height - 440;
    
    // Initialize coins - one above each platform except the ground
    for (int i = 0; i < COIN_COUNT; i++) {
        // Position coin above the corresponding platform
        // i+1 because we skip the ground platform
        Platform* plat = &env->platforms[i+1];
        env->coins[i].x = plat->x + plat->width/2;
        env->coins[i].y = plat->y - 25;  // Position coin 25 units above platform
        env->coins[i].radius = COIN_RADIUS;
        env->coins[i].collected = false;
    }
}

void c_reset(CPlatformer* env) {
    env->player_x = env->width / 2;
    env->player_y = env->height / 2;
    env->player_vx = 0;
    env->player_vy = 0;
    env->player_width = 30;
    env->player_height = 50;
    env->on_ground = false;
    env->tick = 0;
    env->max_ticks = 10000;
    env->rewards[0] = 0;
    env->dones[0] = NOT_DONE;
    env->state = PLAYING;
    
    init_platforms(env);
    compute_observation(env);
}

void finish_game(CPlatformer* env, float reward) {
    env->dones[0] = DONE;
    env->rewards[0] = reward;
    env->log.episode_length = env->tick;
    env->log.episode_return = reward;
    env->log.score = reward;
    add_log(env->log_buffer, &env->log);
}

// Check if all coins have been collected
bool all_coins_collected(CPlatformer* env) {
    for (int i = 0; i < COIN_COUNT; i++) {
        if (!env->coins[i].collected) {
            return false;
        }
    }
    return true;
}

void c_step(CPlatformer* env) {
    if (env->dones[0] == DONE) {
        c_reset(env);
        return;
    }
    
    env->tick++;
    // Don't reset rewards each step, they should accumulate
    // env->rewards[0] = 0;
    
    // Get action
    Action action = env->actions[0];
    
    // Update player position based on action
    if (action == LEFT) {
        env->player_vx = -PLAYER_SPEED;
    } else if (action == RIGHT) {
        env->player_vx = PLAYER_SPEED;
    } else if (action == JUMP && env->on_ground) {
        env->player_vy = -JUMP_FORCE;
        env->on_ground = false;
    } else if (action == NOOP) {
        // No action - don't change velocity
        // Just apply friction and gravity naturally
    }
    
    // Apply gravity
    env->player_vy += GRAVITY;
    
    // Update position
    env->player_x += env->player_vx;
    env->player_y += env->player_vy;
    
    // Boundary checks
    if (env->player_x < 0) {
        env->player_x = 0;
        env->player_vx = 0;
    } else if (env->player_x + env->player_width > env->width) {
        env->player_x = env->width - env->player_width;
        env->player_vx = 0;
    }
    
    // Check platform collisions
    env->on_ground = false;
    for (int i = 0; i < PLATFORM_COUNT; i++) {
        Platform* plat = &env->platforms[i];
        
        // Check if player is standing on this platform
        if (check_standing(env->player_x, env->player_y, env->player_width, env->player_height,
                          plat->x, plat->y, plat->width, plat->height)) {
            env->on_ground = true;
            env->player_y = plat->y - env->player_height;
            env->player_vy = 0;
        }
        // Check collision with platform sides
        else if (check_collision(env->player_x, env->player_y, env->player_width, env->player_height,
                               plat->x, plat->y, plat->width, plat->height)) {
            // Collided with side of platform
            if (env->player_vy > 0) {  // Falling
                env->player_y = plat->y - env->player_height;
                env->player_vy = 0;
                env->on_ground = true;
            } else if (env->player_vy < 0) {  // Jumping up
                env->player_y = plat->y + plat->height;
                env->player_vy = 0;
            }
        }
    }
    
    // Check coin collisions - separate from platform collisions
    for (int i = 0; i < COIN_COUNT; i++) {
        if (!env->coins[i].collected && 
            check_coin_collision(env->player_x, env->player_y, env->player_width, env->player_height,
                                env->coins[i].x, env->coins[i].y, env->coins[i].radius)) {
            // Collect coin and give reward
            env->coins[i].collected = true;
            env->rewards[0] += 1.0 / COIN_COUNT;  // Divide by total coins so collecting all = 1.0
            
            // Check for win condition - all coins collected
            if (all_coins_collected(env)) {
                env->state = WON;
                // Just use accumulated rewards, no special win reward
                finish_game(env, env->rewards[0]);
                return;
            }
        }
    }
    
    // Check fall off screen
    if (env->player_y > env->height) {
        env->state = LOST;
        // No negative reward for falling, just end with current reward
        finish_game(env, env->rewards[0]);
        return;
    }
    
    // Check timeout
    if (env->tick >= env->max_ticks) {
        env->state = LOST;  // Use LOST state for timeout too
        // Apply -1.0 penalty for timeout
        env->rewards[0] -= 1.0;
        finish_game(env, env->rewards[0]);
        return;
    }
    
    // Apply friction if on ground
    if (env->on_ground) {
        env->player_vx *= 0.8;
    }
    
    // Update observation
    compute_observation(env);
}

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D player;
};

Client* make_client(int width, int height) {
    Client* client = (Client*) calloc(1, sizeof(Client));
    client->width = width;
    client->height = height;
    
    InitWindow(width, height, "Platformer");
    SetTargetFPS(60);
    
    // Load player texture
    // Create a blank red square as a placeholder
    Image playerImg = GenImageColor(30, 50, RED);
    client->player = LoadTextureFromImage(playerImg);
    UnloadImage(playerImg);
    
    return client;
}

void c_render(Client* client, CPlatformer* env) {
    BeginDrawing();
    ClearBackground(DARKBLUE);
    
    // Draw platforms
    for (int i = 0; i < PLATFORM_COUNT; i++) {
        if (i == 0) {
            // Ground platform
            DrawRectangle(env->platforms[i].x, env->platforms[i].y, 
                          env->platforms[i].width, env->platforms[i].height, DARKGREEN);
        } else {
            // Regular platforms - all green now (no color change)
            DrawRectangle(env->platforms[i].x, env->platforms[i].y, 
                          env->platforms[i].width, env->platforms[i].height, GREEN);
        }
    }
    
    // Draw coins
    for (int i = 0; i < COIN_COUNT; i++) {
        if (!env->coins[i].collected) {
            DrawCircle(env->coins[i].x, env->coins[i].y, env->coins[i].radius, GOLD);
        }
    }
    
    // Draw player
    DrawTexture(client->player, env->player_x, env->player_y, WHITE);
    
    // Draw info
    DrawText(TextFormat("Score: %.0f", env->rewards[0]), 10, 10, 20, WHITE);
    DrawText(TextFormat("Time: %d/%d", env->tick, env->max_ticks), 10, 40, 20, WHITE);
    
    // Draw win/lose message
    if (env->dones[0] == DONE) {
        const char* message;
        Color messageColor;
        
        if (env->state == WON) {
            message = "YOU WIN!";
            messageColor = GREEN;
        } else if (env->state == LOST) {
            message = "GAME OVER";
            messageColor = RED;
        } else {
            message = "TIME'S UP!";
            messageColor = YELLOW;
        }
        
        int textWidth = MeasureText(message, 40);
        DrawText(message, env->width/2 - textWidth/2, env->height/2 - 20, 40, messageColor);
    }
    
    EndDrawing();
}

void close_client(Client* client) {
    UnloadTexture(client->player);
    CloseWindow();
    free(client);
} 