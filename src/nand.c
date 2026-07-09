#include "nand.h"
#include <stdio.h>
#include <assert.h>

NANDDevice *nand_create(void) {
    NANDDevice *dev = calloc(1, sizeof(NANDDevice));
    if (!dev) return NULL;
    for (uint32_t b = 0; b < NUM_BLOCKS; b++) {
        dev->blocks[b].state      = BLOCK_FREE;
        dev->blocks[b].free_pages = PAGES_PER_BLOCK;
    }
    return dev;
}

void nand_destroy(NANDDevice *dev) { free(dev); }

int nand_read_page(NANDDevice *dev, uint32_t block, uint32_t page,
                   uint8_t *buf) {
    if (block >= NUM_BLOCKS || page >= PAGES_PER_BLOCK) return -1;
    NANDPage *p = &dev->blocks[block].pages[page];
    if (p->state == PAGE_FREE) return -1;
    memcpy(buf, p->data, PAGE_SIZE);
    dev->total_reads++;
    return 0;
}

int nand_write_page(NANDDevice *dev, uint32_t block, uint32_t page,
                    const uint8_t *buf, uint32_t lba) {
    if (block >= NUM_BLOCKS || page >= PAGES_PER_BLOCK) return -1;
    NANDBlock *blk = &dev->blocks[block];
    NANDPage  *p   = &blk->pages[page];

    if (p->state != PAGE_FREE) {
        fprintf(stderr, "NAND: write to non-free page b=%u p=%u\n", block, page);
        return -1;
    }
    if (blk->pe_count >= MAX_PE_CYCLES) return -1; /* worn out */

    memcpy(p->data, buf, PAGE_SIZE);
    p->state     = PAGE_VALID;
    p->lba       = lba;
    p->write_seq = ++dev->write_seq;

    blk->valid_pages++;
    blk->free_pages--;
    if (blk->state == BLOCK_FREE) blk->state = BLOCK_ACTIVE;
    if (blk->free_pages == 0)     blk->state = BLOCK_FULL;

    dev->total_writes++;
    return 0;
}

int nand_erase_block(NANDDevice *dev, uint32_t block) {
    if (block >= NUM_BLOCKS) return -1;
    NANDBlock *blk = &dev->blocks[block];
    if (blk->pe_count >= MAX_PE_CYCLES) return -1;

    memset(blk->pages, 0, sizeof(blk->pages));
    blk->state         = BLOCK_FREE;
    blk->valid_pages   = 0;
    blk->invalid_pages = 0;
    blk->free_pages    = PAGES_PER_BLOCK;
    blk->pe_count++;

    dev->total_erases++;
    return 0;
}

double nand_waf(NANDDevice *dev) {
    if (dev->host_writes == 0) return 0.0;
    return (double)(dev->host_writes + dev->gc_writes) / dev->host_writes;
}
