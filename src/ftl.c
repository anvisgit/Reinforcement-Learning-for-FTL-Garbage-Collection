#include "ftl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

/* ─── internal helpers ─────────────────────────── */

/* find a free block for the write frontier */
static int alloc_free_block(FTL *ftl) {
    /* prefer low PE-count blocks (wear leveling) */
    uint32_t best = UINT32_MAX;
    uint32_t best_pe = UINT32_MAX;
    for (uint32_t b = 0; b < NUM_BLOCKS; b++) {
        if (ftl->nand->blocks[b].state == BLOCK_FREE) {
            uint32_t pe = ftl->nand->blocks[b].pe_count;
            if (pe < best_pe) { best_pe = pe; best = b; }
        }
    }
    return (best == UINT32_MAX) ? -1 : (int)best;
}

/* advance to next write block when current one is full */
static int advance_write_frontier(FTL *ftl) {
    int nb = alloc_free_block(ftl);
    if (nb < 0) return -1;
    ftl->active_block = (uint32_t)nb;
    ftl->active_page  = 0;
    ftl->nand->blocks[nb].state = BLOCK_ACTIVE;
    return 0;
}

/* ─── creation / destruction ───────────────────── */
FTL *ftl_create(NANDDevice *nand, GCPolicy policy) {
    srand((unsigned)time(NULL));
    FTL *ftl = calloc(1, sizeof(FTL));
    if (!ftl) return NULL;
    ftl->nand      = nand;
    ftl->gc_policy = policy;
    for (uint32_t i = 0; i < LBA_COUNT; i++)
        ftl->map[i].block = 0xFFFFFFFF; /* unmapped */

    if (advance_write_frontier(ftl) < 0) { free(ftl); return NULL; }
    return ftl;
}

void ftl_destroy(FTL *ftl) { free(ftl); }

/* ─── read ─────────────────────────────────────── */
int ftl_read(FTL *ftl, uint32_t lba, uint8_t *buf) {
    if (lba >= LBA_COUNT) return -1;
    PPA ppa = ftl->map[lba];
    if (ppa.block == 0xFFFFFFFF) {
        memset(buf, 0, PAGE_SIZE); /* unwritten = zeros */
        return 0;
    }
    return nand_read_page(ftl->nand, ppa.block, ppa.page, buf);
}

/* ─── write ────────────────────────────────────── */
int ftl_write(FTL *ftl, uint32_t lba, const uint8_t *buf) {
    if (lba >= LBA_COUNT) return -1;

    /* invalidate old mapping */
    PPA old = ftl->map[lba];
    if (old.block != 0xFFFFFFFF) {
        NANDBlock *oblk = &ftl->nand->blocks[old.block];
        if (oblk->pages[old.page].state == PAGE_VALID) {
            oblk->pages[old.page].state = PAGE_INVALID;
            oblk->valid_pages--;
            oblk->invalid_pages++;
        }
    }

    /* need GC? (keep ≥ OP_BLOCKS free as breathing room) */
    uint32_t free_blocks = 0;
    for (uint32_t b = 0; b < NUM_BLOCKS; b++)
        if (ftl->nand->blocks[b].state == BLOCK_FREE) free_blocks++;
    if (free_blocks < OP_BLOCKS / 2 + 1)
        ftl_gc(ftl);

    /* need a new write block? */
    if (ftl->active_page >= PAGES_PER_BLOCK) {
        if (advance_write_frontier(ftl) < 0) {
            ftl_gc(ftl);
            if (advance_write_frontier(ftl) < 0) return -1;
        }
    }

    int rc = nand_write_page(ftl->nand, ftl->active_block,
                             ftl->active_page, buf, lba);
    if (rc < 0) return -1;

    ftl->map[lba].block = ftl->active_block;
    ftl->map[lba].page  = ftl->active_page;
    ftl->active_page++;
    ftl->nand->host_writes++;
    return 0;
}

/* ─── GC candidate selection ───────────────────── */

/* collect candidate blocks (not active, not free) */
static uint32_t collect_candidates(FTL *ftl, uint32_t *cand, uint32_t max) {
    uint32_t n = 0;
    for (uint32_t b = 0; b < NUM_BLOCKS && n < max; b++) {
        NANDBlock *blk = &ftl->nand->blocks[b];
        if (b == ftl->active_block)   continue;
        if (blk->state == BLOCK_FREE) continue;
        /* must have at least one invalid page to be worth erasing,
           and must not be the currently-filling write frontier */
        if (blk->invalid_pages == 0) continue;
        cand[n++] = b;
    }
    return n;
}

static uint32_t gc_greedy(FTL *ftl, uint32_t *cand, uint32_t n) {
    uint32_t best = cand[0], best_inv = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t inv = ftl->nand->blocks[cand[i]].invalid_pages;
        if (inv > best_inv) { best_inv = inv; best = cand[i]; }
    }
    return best;
}

static uint32_t gc_random(uint32_t *cand, uint32_t n) {
    return cand[rand() % n];
}

/* ─── state vector (for RL) ─────────────────────
   Features per candidate block (up to 32 candidates, 5 features each):
   [invalid_ratio, pe_count_norm, valid_ratio, free_ratio, age_norm]
   Plus global features: [waf, util, free_ratio_global, gc_pressure]
   Total ≤ 32*5 + 4 = 164 features; we trim to actual n_cand.
*/
int ftl_state_vector(FTL *ftl, float *features, int max_features,
                     uint32_t *cand_blocks, uint32_t *n_cand) {
    uint32_t cand[NUM_BLOCKS];
    uint32_t n = collect_candidates(ftl, cand, NUM_BLOCKS);
    if (n > 32) n = 32;
    *n_cand = n;
    memcpy(cand_blocks, cand, n * sizeof(uint32_t));

    uint32_t free_cnt = 0;
    for (uint32_t b = 0; b < NUM_BLOCKS; b++)
        if (ftl->nand->blocks[b].state == BLOCK_FREE) free_cnt++;

    int idx = 0;
    /* global features */
    if (idx < max_features) features[idx++] = (float)ftl_waf(ftl);
    if (idx < max_features) features[idx++] = (float)(NUM_BLOCKS - free_cnt) / NUM_BLOCKS;
    if (idx < max_features) features[idx++] = (float)free_cnt / NUM_BLOCKS;
    double gc_pressure = (free_cnt < OP_BLOCKS) ? 1.0 : 0.0;
    if (idx < max_features) features[idx++] = (float)gc_pressure;

    for (uint32_t i = 0; i < n && idx + 5 <= max_features; i++) {
        NANDBlock *blk = &ftl->nand->blocks[cand[i]];
        features[idx++] = (float)blk->invalid_pages / PAGES_PER_BLOCK;
        features[idx++] = (float)blk->valid_pages   / PAGES_PER_BLOCK;
        features[idx++] = (float)blk->free_pages    / PAGES_PER_BLOCK;
        features[idx++] = (float)blk->pe_count      / MAX_PE_CYCLES;
        /* relative age: write_seq of oldest page in block */
        uint64_t oldest = UINT64_MAX;
        for (uint32_t p = 0; p < PAGES_PER_BLOCK; p++)
            if (blk->pages[p].state == PAGE_VALID &&
                blk->pages[p].write_seq < oldest)
                oldest = blk->pages[p].write_seq;
        float age = (oldest == UINT64_MAX) ? 0.0f
                  : (float)(ftl->nand->write_seq - oldest) /
                    (float)(ftl->nand->write_seq + 1);
        features[idx++] = age;
    }
    return idx;
}

/* ─── GC ────────────────────────────────────────── */
int ftl_gc(FTL *ftl) {
    uint32_t cand[NUM_BLOCKS];
    uint32_t n = collect_candidates(ftl, cand, NUM_BLOCKS);
    if (n == 0) return -1;

    uint32_t victim;
    if (ftl->gc_policy == GC_POLICY_GREEDY) {
        victim = gc_greedy(ftl, cand, n);
    } else if (ftl->gc_policy == GC_POLICY_RANDOM) {
        victim = gc_random(cand, n);
    } else {
        /* RL policy: delegate to callback if set, else fall back to greedy */
        if (ftl->rl_pick_victim) {
            float features[200];
            uint32_t cand2[NUM_BLOCKS];
            uint32_t nc = 0;
            ftl_state_vector(ftl, features, 200, cand2, &nc);
            victim = ftl->rl_pick_victim(ftl->rl_ctx, cand2, nc, features);
        } else {
            victim = gc_greedy(ftl, cand, n);
        }
    }

    NANDBlock *vblk = &ftl->nand->blocks[victim];

    /* dense reward for the RL path: reward is the fraction of invalid
       pages in the chosen victim block (1.0 = perfect, all garbage;
       0.0 = wasteful, all valid data had to be copied). This is
       action-dependent and computed *after* the victim is known, unlike
       a lagging global WAF delta. */
    if (ftl->gc_policy == GC_POLICY_RL && ftl->rl_pick_victim) {
        double inv_ratio = (double)vblk->invalid_pages / PAGES_PER_BLOCK;
        ftl->last_gc_reward = inv_ratio; /* consumed by caller on next IPC round */
    }

    /* relocate valid pages */
    for (uint32_t p = 0; p < PAGES_PER_BLOCK; p++) {
        NANDPage *pg = &vblk->pages[p];
        if (pg->state != PAGE_VALID) continue;

        uint32_t lba = pg->lba;
        /* verify this page still owns lba */
        if (ftl->map[lba].block != victim || ftl->map[lba].page != p)
            continue;

        /* need new write slot */
        if (ftl->active_page >= PAGES_PER_BLOCK) {
            if (advance_write_frontier(ftl) < 0) return -1;
        }

        int rc = nand_write_page(ftl->nand, ftl->active_block,
                                 ftl->active_page, pg->data, lba);
        if (rc < 0) return -1;

        ftl->map[lba].block = ftl->active_block;
        ftl->map[lba].page  = ftl->active_page;
        ftl->active_page++;
        ftl->nand->gc_writes++;
        ftl->total_gc_copies++;
    }

    nand_erase_block(ftl->nand, victim);
    ftl->total_gc_runs++;
    return (int)victim;
}

/* ─── stats ─────────────────────────────────────── */
double ftl_waf(FTL *ftl) { return nand_waf(ftl->nand); }

void ftl_print_stats(FTL *ftl) {
    NANDDevice *d = ftl->nand;
    printf("=== FTL Stats ===\n");
    printf("  Host writes  : %lu\n",  (unsigned long)d->host_writes);
    printf("  GC  writes   : %lu\n",  (unsigned long)d->gc_writes);
    printf("  Total writes : %lu\n",  (unsigned long)d->total_writes);
    printf("  Total erases : %lu\n",  (unsigned long)d->total_erases);
    printf("  GC runs      : %lu\n",  (unsigned long)ftl->total_gc_runs);
    printf("  GC copies    : %lu\n",  (unsigned long)ftl->total_gc_copies);
    printf("  WAF          : %.4f\n", ftl_waf(ftl));
    uint32_t free_b = 0, used_b = 0;
    for (uint32_t b = 0; b < NUM_BLOCKS; b++)
        (d->blocks[b].state == BLOCK_FREE) ? free_b++ : used_b++;
    printf("  Free blocks  : %u / %u\n", free_b, NUM_BLOCKS);
}
