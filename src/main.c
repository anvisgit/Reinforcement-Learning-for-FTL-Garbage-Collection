#include "nand.h"
#include "ftl.h"
#include "ipc_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <errno.h>

#if defined(_WIN32)
#include <direct.h>
#define mkdir(path) _mkdir(path)
#else
#include <sys/stat.h>
#endif

/* ─── workload generator ────────────────────────── */
typedef enum { WL_SEQ, WL_RANDOM, WL_HOTCOLD } WorkloadType;

static void gen_lba(WorkloadType wl, uint32_t step,
                    uint32_t *lba_out, uint8_t *buf_out) {
    uint32_t lba;
    if (wl == WL_SEQ) {
        lba = step % LBA_COUNT;
    } else if (wl == WL_RANDOM) {
        lba = rand() % LBA_COUNT;
    } else { /* hot-cold: 20% of addresses get 80% of writes */
        if ((rand() % 100) < 80)
            lba = rand() % (LBA_COUNT / 5);
        else
            lba = LBA_COUNT / 5 + rand() % (LBA_COUNT * 4 / 5);
    }
    *lba_out = lba;
    /* fill page data with identifiable pattern */
    for (int i = 0; i < (int)PAGE_SIZE; i++)
        buf_out[i] = (uint8_t)((lba + i + step) & 0xFF);
}

/* ─── WAF record at intervals ───────────────────── */
#define RECORD_INTERVAL 1000
#define MAX_RECORDS     200

typedef struct {
    uint64_t host_writes;
    double   waf;
    uint64_t gc_runs;
} Record;

/* ─── RL victim picker via IPC ──────────────────── */

static uint32_t rl_pick_via_ipc(void *ctx, uint32_t *cand, uint32_t n_cand,
                                  float *features) {
    FTL *ftl = *(FTL **)ctx;

    /* reward for the *previous* GC decision: dense, action-dependent
       (fraction of invalid pages in the block we just erased). On the
       very first call there is no previous decision, so reward = 0. */
    double reward = ftl->last_gc_reward;

    /* exact feature count: 4 global + 5 per candidate (see ftl_state_vector) */
    uint32_t n_feat = 4 + n_cand * 5;
    if (n_feat > IPC_MAX_FEAT) n_feat = IPC_MAX_FEAT;

    return ipc_call_agent(features, n_feat < 4 ? 4 : n_feat,
                          cand, n_cand, reward, 0);
}

/* ─── run one benchmark ─────────────────────────── */
typedef struct {
    Record   records[MAX_RECORDS];
    uint32_t n_records;
    double   final_waf;
    uint64_t total_gc_runs;
    uint64_t total_gc_copies;
    char     name[32];
} BenchResult;

static BenchResult run_bench(GCPolicy policy, WorkloadType wl,
                              uint32_t n_writes, const char *name,
                              bool use_ipc) {
    BenchResult res = {0};
    strncpy(res.name, name, 31);

    NANDDevice *nand = nand_create();
    FTL        *ftl  = ftl_create(nand, policy);

    FTL *ftl_ptr = ftl;
    if (use_ipc) {
        ftl->gc_policy      = GC_POLICY_RL;
        ftl->rl_ctx         = &ftl_ptr;
        ftl->rl_pick_victim = rl_pick_via_ipc;
    }

    uint8_t buf[PAGE_SIZE];
    uint64_t max_gc_runs = use_ipc ? (uint64_t)n_writes * 4 : UINT64_MAX;

    for (uint32_t i = 0; i < n_writes; i++) {
        uint32_t lba;
        gen_lba(wl, i, &lba, buf);
        ftl_write(ftl, lba, buf);

        if (use_ipc && ftl->total_gc_runs > max_gc_runs) {
            fprintf(stderr, "  [safety] GC run cap hit, ending episode early at write %u\n", i);
            break;
        }

        if (i % RECORD_INTERVAL == 0 && res.n_records < MAX_RECORDS) {
            res.records[res.n_records].host_writes = nand->host_writes;
            res.records[res.n_records].waf         = ftl_waf(ftl);
            res.records[res.n_records].gc_runs     = ftl->total_gc_runs;
            res.n_records++;
        }
    }

    res.final_waf     = ftl_waf(ftl);
    res.total_gc_runs  = ftl->total_gc_runs;
    res.total_gc_copies= ftl->total_gc_copies;

    if (use_ipc) {
        /* signal done */
        float dummy[4] = {0};
        uint32_t dc = 0;
        ipc_call_agent(dummy, 4, &dc, 1, 0.0, 1);
    }

    ftl_print_stats(ftl);
    ftl_destroy(ftl);
    nand_destroy(nand);
    return res;
}

/* ─── save results to CSV ───────────────────────── */
static void save_csv(BenchResult *results, int n, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { perror("fopen"); return; }
    fprintf(f, "policy,host_writes,waf,gc_runs\n");
    for (int r = 0; r < n; r++) {
        for (uint32_t i = 0; i < results[r].n_records; i++) {
            fprintf(f, "%s,%lu,%.6f,%lu\n",
                results[r].name,
                (unsigned long)results[r].records[i].host_writes,
                results[r].records[i].waf,
                (unsigned long)results[r].records[i].gc_runs);
        }
    }
    fclose(f);
    printf("Saved: %s\n", path);
}

static void save_summary(BenchResult *results, int n, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "policy,final_waf,total_gc_runs,total_gc_copies\n");
    for (int r = 0; r < n; r++) {
        fprintf(f, "%s,%.6f,%lu,%lu\n",
            results[r].name,
            results[r].final_waf,
            (unsigned long)results[r].total_gc_runs,
            (unsigned long)results[r].total_gc_copies);
    }
    fclose(f);
    printf("Saved: %s\n", path);
}

/* ─── main ──────────────────────────────────────── */
int main(int argc, char **argv) {
    bool use_rl  = (argc > 1 && strcmp(argv[1], "--rl") == 0);
    bool train   = (argc > 2 && strcmp(argv[2], "--train") == 0);
    (void)train;

    /* workload */
    WorkloadType wl      = WL_HOTCOLD;
    uint32_t     n_writes = LBA_COUNT * 3;  /* 3× full drive writes */
    const char *nw_env = getenv("FTL_N_WRITES");
    if (nw_env) n_writes = (uint32_t)atoi(nw_env);

    printf("=== FTL-RL Benchmark ===\n");
    printf("LBA_COUNT=%u  N_WRITES=%u  WORKLOAD=hot-cold\n\n",
           LBA_COUNT, n_writes);

    BenchResult results[3];
    int         n_res = 0;

    bool skip_baselines = (getenv("FTL_SKIP_BASELINES") != NULL);

    if (!skip_baselines) {
    /* 1. Greedy */
    printf("--- Greedy GC ---\n");
    results[n_res++] = run_bench(GC_POLICY_GREEDY, wl, n_writes,
                                  "greedy", false);
    printf("\n");

    /* 2. Random */
    printf("--- Random GC ---\n");
    results[n_res++] = run_bench(GC_POLICY_RANDOM, wl, n_writes,
                                  "random", false);
    printf("\n");
    }

    /* 3. RL (only if --rl flag given; agent must be running separately) */
    if (use_rl) {
        printf("--- RL (PPO) GC ---\n");
        if (ipc_open_server() < 0) {
            fprintf(stderr, "IPC init failed\n");
            return 1;
        }
        results[n_res++] = run_bench(GC_POLICY_RL, wl, n_writes,
                                      "ppo", true);
        ipc_close_server();
        printf("\n");
    }

    /* save results */
    if (n_res > 0) {
        if (mkdir("results") != 0 && errno != EEXIST) {
            perror("mkdir");
        }
        save_csv    (results, n_res, "results/waf_curves.csv");
        save_summary(results, n_res, "results/summary.csv");
    }

    printf("\n=== Summary ===\n");
    for (int i = 0; i < n_res; i++)
        printf("  %-8s  WAF=%.4f  GC_runs=%lu\n",
            results[i].name,
            results[i].final_waf,
            (unsigned long)results[i].total_gc_runs);

    return 0;
}
