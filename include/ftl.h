#ifndef FTL_H
#define FTL_H

#include "nand.h"
#include <stdint.h>
#include <stdbool.h>

/* ──────────────── address mapping ──────────────── */
/* Page-level mapping table: lba -> (block, page)   */
#define LBA_COUNT  (USER_BLOCKS * PAGES_PER_BLOCK)

typedef struct {
    uint32_t block;   /* 0xFFFFFFFF = unmapped */
    uint32_t page;
} PPA;                /* Physical Page Address */

/* ──────────────── GC policy tag ─────────────────  */
typedef enum {
    GC_POLICY_GREEDY  = 0,   /* pick block with most invalid pages */
    GC_POLICY_RANDOM  = 1,   /* pick a random candidate            */
    GC_POLICY_RL      = 2,   /* PPO agent (implemented in Python)  */
} GCPolicy;

/* ──────────────── FTL context ──────────────────── */
typedef struct {
    NANDDevice *nand;
    PPA         map[LBA_COUNT];      /* L2P mapping                  */

    uint32_t    active_block;        /* current write frontier block */
    uint32_t    active_page;         /* next free page in that block */

    uint64_t    total_gc_runs;
    uint64_t    total_gc_copies;     /* valid pages relocated         */

    GCPolicy    gc_policy;

    /* stats ring for RL state vector */
    double      waf_history[16];
    uint32_t    waf_idx;
    double      last_gc_reward;   /* dense reward for most recent GC decision */

    /* callback: RL agent picks victim block (set to NULL for built-ins) */
    uint32_t (*rl_pick_victim)(void *ctx, uint32_t *candidates,
                               uint32_t n_cand, float *features);
    void     *rl_ctx;
} FTL;

/* ──────────────── API ─────────────────────────── */
FTL  *ftl_create(NANDDevice *nand, GCPolicy policy);
void  ftl_destroy(FTL *ftl);

int   ftl_write(FTL *ftl, uint32_t lba, const uint8_t *buf);
int   ftl_read (FTL *ftl, uint32_t lba, uint8_t *buf);

/* Trigger one GC cycle; returns victim block id or -1 if nothing to do */
int   ftl_gc(FTL *ftl);

/* State vector for RL (fills feature array, returns length) */
int   ftl_state_vector(FTL *ftl, float *features, int max_features,
                       uint32_t *cand_blocks, uint32_t *n_cand);

/* Helpers */
double ftl_waf(FTL *ftl);
void   ftl_print_stats(FTL *ftl);

#endif /* FTL_H */
