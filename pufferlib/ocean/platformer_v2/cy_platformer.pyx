cimport numpy as cnp
from libc.stdlib cimport calloc, free
import os

cdef extern from "platformer.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return
        float episode_length
        float success_rate

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Player:
        float x
        float y
        float vel_x
        float vel_y
        int width
        int height
        int on_ground

    ctypedef struct Goal:
        float x
        float y
        int width
        int height

    ctypedef struct Enemy:
        float x
        float y
        int width
        int height

    ctypedef struct Platform:
        float x
        float y
        int width
        int height

    ctypedef struct CPlatformer:
        float* observations
        int* actions
        float* rewards
        unsigned char* dones
        LogBuffer* log_buffer
        Log log
        
        Player player
        Goal goal
        Enemy enemy
        Platform ground
        
        int steps
        float episode_return
        
        int screen_width
        int screen_height

    ctypedef struct Client
    void free_cplatformer(CPlatformer* env)
    Client* make_client(int width, int height)
    void close_client(Client* client)
    void c_render(Client* client, CPlatformer* env)
    void c_reset(CPlatformer* env)
    void c_step(CPlatformer* env)

cdef class CyPlatformer:
    cdef:
        CPlatformer* envs
        Client* client
        LogBuffer* logs
        int num_envs

    def __init__(self, float[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs,
            int width, int height, int player_size):

        self.num_envs = num_envs
        self.client = NULL
        self.envs = <CPlatformer*> calloc(num_envs, sizeof(CPlatformer))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int i
        for i in range(num_envs):
            self.envs[i] = CPlatformer(
                observations=&observations[i, 0],
                actions=&actions[i],
                rewards=&rewards[i],
                dones=&terminals[i],
                log_buffer=self.logs,
                player=Player(
                    x=50,
                    y=height - player_size - 20,
                    vel_x=0,
                    vel_y=0,
                    width=player_size,
                    height=player_size,
                    on_ground=1
                ),
                goal=Goal(
                    x=width - 100,
                    y=height - player_size - 20,
                    width=player_size,
                    height=player_size
                ),
                enemy=Enemy(
                    x=width / 2,  # Place enemy in the middle
                    y=height - player_size - 20,
                    width=player_size,
                    height=player_size
                ),
                ground=Platform(
                    x=0,
                    y=height - 20,
                    width=width,
                    height=20
                ),
                steps=0,
                episode_return=0,
                screen_width=width,
                screen_height=height
            )

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        cdef CPlatformer* env = &self.envs[0]
        if self.client == NULL:
            import os
            cwd = os.getcwd()
            os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            self.client = make_client(env.screen_width, env.screen_height)
            os.chdir(cwd)

        c_render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        # Free resources
        free_cplatformer(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return {
            'episode_return': log.episode_return,
            'episode_length': log.episode_length,
            'success_rate': log.success_rate
        } 