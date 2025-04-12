#include "platformer.h"
#include "time.h"

void interactive() {
    CPlatformer env = {
        .width = 800,
        .height = 600,
        .player_width = 30,
        .player_height = 50,
    };
    allocate_cplatformer(&env);
    c_reset(&env);
 
    Client* client = make_client(env.width, env.height);
    
    int tick = 0;
    while (!WindowShouldClose()) {
        env.actions[0] = NOOP;  // Default to no action
        
        // Handle user input
        if (IsKeyDown(KEY_LEFT)) env.actions[0] = LEFT;
        if (IsKeyDown(KEY_RIGHT)) env.actions[0] = RIGHT;
        if (IsKeyPressed(KEY_SPACE)) env.actions[0] = JUMP;
        
        c_step(&env);
        c_render(client, &env);
    }
    
    close_client(client);
    free_allocated_cplatformer(&env);
}

void performance_test() {
    long test_time = 10;
    CPlatformer env = {
        .width = 800,
        .height = 600,
        .player_width = 30,
        .player_height = 50,
    };
    allocate_cplatformer(&env);
    c_reset(&env);
 
    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand() % ACTIONS_COUNT;
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated_cplatformer(&env);
}

int main() {
    //performance_test();
    interactive();
    return 0;
} 