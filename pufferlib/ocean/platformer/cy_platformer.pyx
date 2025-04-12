cimport numpy as cnp
from libc.stdlib cimport calloc, free
import os

cdef extern from "platformer.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return;
        float episode_length;
        float score;

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct CPlatformer:
        float* observations
        int* actions
        float* rewards
        unsigned char* dones
        LogBuffer* log_buffer;
        Log log;

        float player_x;
        float player_y;
        float player_vx;
        float player_vy;
        float player_width;
        float player_height;
        bint on_ground;
        
        int tick;
        int max_ticks;
        
        int width;
        int height;

    ctypedef struct Client
    void free_cplatformer(CPlatformer* env)
    Client* make_client(float width, float height)
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
            int width, int height, int player_width, int player_height):

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
                player_x=0,
                player_y=0,
                player_vx=0,
                player_vy=0,
                player_width=player_width,
                player_height=player_height,
                on_ground=False,
                tick=0,
                max_ticks=10000,
                width=width,
                height=height,
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
            self.client = make_client(env.width, env.height)
            os.chdir(cwd)

        c_render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return {'episode_return': log.episode_return, 
                'episode_length': log.episode_length,
                'score': log.score} 