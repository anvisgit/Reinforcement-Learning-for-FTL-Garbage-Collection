#ifndef NAND_H
#define NAND_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ──────────────── NAND geometry ──────────────── */
#define PAGE_SIZE        4096      /* bytes per page          */
#define PAGES_PER_BLOCK  64        /* pages per erase block   */
#define NUM_BLOCKS       256       /* total erase blocks      */
#define MAX_PE_CYCLES    10000     /* program/erase endurance */

#define BLOCK_SIZE  (PAGE_SIZE * PAGES_PER_BLOCK)
#define TOTAL_PAGES (NUM_BLOCKS * PAGES_PER_BLOCK)

/* over-provisioning: top 20% reserved for FTL metadata / GC */
#define USER_BLOCKS     (NUM_BLOCKS * 80 / 100)
#define OP_BLOCKS       (NUM_BLOCKS - USER_BLOCKS)

/* ──────────────── Page / Block state ──────────── */
typedef enum { PAGE_FREE = 0, PAGE_VALID, PAGE_INVALID } PageState;
typedef enum { BLOCK_FREE = 0, BLOCK_ACTIVE, BLOCK_FULL, BLOCK_GC_VICTIM } BlockState;

typedef struct {
    uint8_t      data[PAGE_SIZE];
    PageState    state;
    uint32_t     lba;           /* logical block address written here */
    uint64_t     write_seq;     /* monotonic write counter            */
} NANDPage;

typedef struct {
    NANDPage     pages[PAGES_PER_BLOCK];
    BlockState   state;
    uint32_t     pe_count;      /* program/erase cycles               */
    uint32_t     valid_pages;
    uint32_t     invalid_pages;
    uint32_t     free_pages;
} NANDBlock;

typedef struct {
    NANDBlock    blocks[NUM_BLOCKS];
    uint64_t     total_reads;
    uint64_t     total_writes;       /* physical page writes            */
    uint64_t     total_erases;
    uint64_t     host_writes;        /* logical writes from host        */
    uint64_t     gc_writes;          /* writes caused by GC             */
    uint64_t     write_seq;          /* global monotonic counter        */
} NANDDevice;

/* ──────────────── API ─────────────────────────── */
NANDDevice *nand_create(void);
void        nand_destroy(NANDDevice *dev);

/* returns 0 on success, -1 on failure (bad block / over PE) */
int  nand_read_page (NANDDevice *dev, uint32_t block, uint32_t page,
                     uint8_t *buf);
int  nand_write_page(NANDDevice *dev, uint32_t block, uint32_t page,
                     const uint8_t *buf, uint32_t lba);
int  nand_erase_block(NANDDevice *dev, uint32_t block);

/* convenience getters */
static inline uint32_t nand_valid_pages(NANDDevice *dev, uint32_t b)
    { return dev->blocks[b].valid_pages; }
static inline uint32_t nand_invalid_pages(NANDDevice *dev, uint32_t b)
    { return dev->blocks[b].invalid_pages; }
static inline uint32_t nand_free_pages(NANDDevice *dev, uint32_t b)
    { return dev->blocks[b].free_pages; }
static inline uint32_t nand_pe_count(NANDDevice *dev, uint32_t b)
    { return dev->blocks[b].pe_count; }

double nand_waf(NANDDevice *dev);   /* write amplification factor */

#endif /* NAND_H */
