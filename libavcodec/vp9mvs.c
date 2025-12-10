/*
 * VP9 compatible video decoder
 *
 * Copyright (C) 2013 Ronald S. Bultje <rsbultje gmail com>
 * Copyright (C) 2013 Clément Bœsch <u pkh me>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "progressframe.h"
#include "vp89_rac.h"
#include "vp9data.h"
#include "vp9dec.h"
#include "vpx_rac.h"
#include "vp9shared.h" // videoparser
#include "libavutil/frame.h" // videoparser
#include <math.h> // videoparser

// videoparser
#define SQR(_x_) ((_x_) * (_x_))

// videoparser: Block size to 4x4 count lookup table
// Index by BlockSize enum, value is number of 4x4 blocks in that block size
// Calculated from ff_vp9_bwh_tab[0][bs][0] * ff_vp9_bwh_tab[0][bs][1]
static const int vp9_bs_to_4x4_count[N_BS_SIZES] = {
    256,  // BS_64x64: 16*16
    128,  // BS_64x32: 16*8
    128,  // BS_32x64: 8*16
     64,  // BS_32x32: 8*8
     32,  // BS_32x16: 8*4
     32,  // BS_16x32: 4*8
     16,  // BS_16x16: 4*4
      8,  // BS_16x8:  4*2
      8,  // BS_8x16:  2*4
      4,  // BS_8x8:   2*2
      2,  // BS_8x4:   2*1
      2,  // BS_4x8:   1*2
      1,  // BS_4x4:   1*1
};

// videoparser: POC-based motion vector normalization flag for VP9
// When enabled, replicates legacy parser behavior for compatibility
// See DEVELOPERS.md for details on the legacy implementation
#ifndef VP_MV_POC_NORMALIZATION
#define VP_MV_POC_NORMALIZATION 0
#endif

// videoparser: Debug flag for investigating frame distance calculation
// Enable with: -DVP_MV_DEBUG_FRMDIST=1
#ifndef VP_MV_DEBUG_FRMDIST
#define VP_MV_DEBUG_FRMDIST 0
#endif

#if VP_MV_DEBUG_FRMDIST
#include <stdio.h>
#include <inttypes.h>
static int vp9_debug_frame_count = 0;
static int64_t vp9_debug_last_pts = INT64_MIN;
static double vp9_debug_frmdist_sum = 0.0;
static int vp9_debug_frmdist_count = 0;
static int vp9_debug_checked_init = 0;  // Track if we've checked init for this frame
static int vp9_debug_ref_count[3] = {0, 0, 0};  // Count blocks by reference type (LAST, GOLDEN, ALTREF)
#endif

/**
 * Motion vector statistics extraction for VP9.
 *
 * When VP_MV_POC_NORMALIZATION=0 (default):
 *   - Extracts MV values without any normalization
 *   - Motion vectors are in 1/8 pel units (VP9 native precision)
 *   - Stats accumulated for all inter modes (NEARESTMV, NEARMV, NEWMV)
 *
 * When VP_MV_POC_NORMALIZATION=1 (legacy mode):
 *   - Normalizes MVs by temporal distance: mv / (8 * FrmDist)
 *   - Applies 4x multiplier to MV lengths: MV_Length = 4.0 * sqrt(...)
 *   - Only accumulates stats for NEWMV mode blocks
 *   - Uses block-size weighted counting (count = number of 4x4 blocks)
 *   - Uses count*count weighting for variance (legacy behavior)
 *   - Applies outlier rejection: blocks where MV > 20x running average are rejected
 *
 * @param sf SharedFrameInfo to accumulate statistics into
 * @param mv Motion vector array [0]=L0, [1]=L1
 * @param comp Whether this is compound (bi-predictive) mode
 * @param mvd_x MVD x component (only valid for NEWMV mode)
 * @param mvd_y MVD y component (only valid for NEWMV mode)
 * @param is_newmv Whether this is NEWMV mode (explicitly coded MV)
 * @param frm_dist Frame distance for POC normalization (only used when VP_MV_POC_NORMALIZATION=1)
 * @param bs Block size enum for this block
 * @param sb Sub-block index (-1 for full block, 0-3 for sub-blocks)
 * @param coded_mv_cnt Number of non-zero MV joints coded (0, 1, or 2 for compound) - legacy mode only
 */
static void mv_statistics_vp9(SharedFrameInfo *sf, const VP9mv *mv,
                              int comp, int mvd_x, int mvd_y, int is_newmv,
                              double frm_dist, enum BlockSize bs, int sb,
                              int is_skip, int coded_mv_cnt) {
    double mv_x = 0.0, mv_y = 0.0;
    double mvd_len = 0.0;
    double mv_length_xy;
    int count = 1;  // Default: 1 unit per MV
#if VP_MV_POC_NORMALIZATION
    double mvd_x_norm = 0.0, mvd_y_norm = 0.0;
    // LEGACY BUG REPLICATION: AvMot and AvDif are integers in legacy code
    // (VideoStatVP9.c line 198), causing truncation when dividing doubles
    int av_mot = 1, av_dif = 1;

    // Legacy mode: skip blocks go to NumBlksSkip, not NumBlksMv
    // We don't track NumBlksSkip, but we need to NOT count them in mb_mv_count
    if (is_skip) {
        return;  // Skip blocks don't contribute to NumBlksMv
    }

    // Legacy mode: use block-size based counting
    // For sub-blocks (sb >= 0): count=1 (each sub-block is one 4x4 unit)
    // For full blocks (sb == -1): count = block_size in 4x4 units
    if (sb == -1) {
        // Full block - use the 4x4 count for this block size
        count = vp9_bs_to_4x4_count[bs];
    }
    // Otherwise count=1 (sub-block)
#else
    (void)frm_dist;      // Unused in non-legacy mode
    (void)bs;            // Unused in non-legacy mode
    (void)sb;            // Unused in non-legacy mode
    (void)is_skip;       // Unused in non-legacy mode
    (void)coded_mv_cnt;  // Unused in non-legacy mode
#endif

    // Always count inter blocks (this is NumBlksMv in legacy)
    // Legacy counts ALL non-skip inter blocks here, not just NEWMV
    sf->mb_mv_count += count;

#if VP_MV_POC_NORMALIZATION
    // Legacy mode: only accumulate MV stats for NEWMV blocks
    // But mb_mv_count (NumBlksMv) is used as denominator for ALL inter blocks
    if (!is_newmv) {
        return;  // Count was incremented above, but don't accumulate MV values
    }
#endif

    // L0 reference (always present for inter blocks)
    mv_x = fabs((double)mv[0].x);
    mv_y = fabs((double)mv[0].y);

    // Note: Legacy VP9 implementation did NOT average compound modes
    // (the code was commented out), so we don't do it either for consistency
    (void)comp;  // Suppress unused warning

#if VP_MV_POC_NORMALIZATION
    // Legacy normalization: divide by 8 * frame_distance
    // The factor of 8 converts from 1/8 pel units to full pixels
    mv_x /= (8.0 * frm_dist);
    mv_y /= (8.0 * frm_dist);

    // MVD also normalized
    mvd_x_norm = fabs((double)mvd_x) / (8.0 * frm_dist);
    mvd_y_norm = fabs((double)mvd_y) / (8.0 * frm_dist);

    // Legacy applies 4x multiplier to MV lengths
    if (mv_x != 0.0 || mv_y != 0.0) {
        mv_length_xy = 4.0 * sqrt(SQR(mv_x) + SQR(mv_y));
        mvd_len = 4.0 * sqrt(SQR(mvd_x_norm) + SQR(mvd_y_norm));
    } else {
        mv_length_xy = 0.0;
        mvd_len = 0.0;
    }

    // LEGACY OUTLIER REJECTION: The legacy VideoStatVP9.c (ProcessMV function)
    // rejects blocks where the normalized MV component sum exceeds 20x the
    // running average. This computes running averages from accumulated values
    // BEFORE adding the current block:
    //   AvMot = MV_Length / CodedMv  (average of 4x-multiplied MV lengths)
    //   AvDif = MV_dLength / CodedMv (average of 4x-multiplied MVD lengths)
    // If CodedMv is 0 (first block), defaults to 1
    // Outlier check: (abs(mvX) + abs(mvY) > 20 * AvMot) || (abs(mvdX) + abs(mvdY) > 20 * AvDif)
    //
    // LEGACY BUG REPLICATION: AvMot and AvDif are integers, causing truncation.
    // Also, abs() in C is for integers - when applied to doubles it truncates first.
    // So abs(5.7) becomes abs(5) = 5.
    if (sf->mv_coded_count > 0) {
        av_mot = (int)(sf->mv_length / sf->mv_coded_count);
        av_dif = (int)(sf->mv_length_diff / sf->mv_coded_count);
    }
    // else av_mot and av_dif remain at 1 (legacy default)

    // Check if this block is an outlier - if so, don't accumulate MV stats
    // LEGACY BUG REPLICATION: abs() on doubles truncates to int first
    // Legacy code: abs(mvX) + abs(mvY) > 20 * AvMot
    // Since mvX/mvY are already positive (from fabs earlier), we just cast to int
    if (((int)mv_x + (int)mv_y > 20 * av_mot) ||
        ((int)mvd_x_norm + (int)mvd_y_norm > 20 * av_dif)) {
        // Outlier detected - skip accumulation but mb_mv_count was already incremented
        return;
    }

    // Not an outlier - accumulate stats
    // LEGACY: CodedMv += b->CodedMv[idx] * count, where b->CodedMv[idx] is
    // the number of non-zero MV joints (0, 1, or 2 for compound mode)
    // This is passed as coded_mv_cnt parameter
    sf->mv_coded_count += coded_mv_cnt * count;

    // Legacy accumulates values multiplied by count
    // and squares multiplied by count*count
    sf->mv_length += mv_length_xy * count;
    sf->mv_sum_sqr += SQR(mv_length_xy) * count * count;
    sf->mv_x_length += mv_x * count;
    sf->mv_y_length += mv_y * count;
    sf->mv_x_sum_sqr += SQR(mv_x) * count * count;
    sf->mv_y_sum_sqr += SQR(mv_y) * count * count;
    sf->mv_length_diff += mvd_len * count;
    sf->mv_diff_sum_sqr += SQR(mvd_len) * count * count;
#else
    // Non-legacy mode: raw values without normalization
    mv_length_xy = sqrt(SQR(mv_x) + SQR(mv_y));

    // For diff stats, use the coded MVD if available (NEWMV mode)
    if (is_newmv) {
        mvd_len = sqrt(SQR((double)mvd_x) + SQR((double)mvd_y));
    }

    // Accumulate statistics (only for NEWMV in legacy mode, all modes otherwise)
    sf->mv_length += mv_length_xy;
    sf->mv_sum_sqr += SQR(mv_length_xy);
    sf->mv_x_length += mv_x;
    sf->mv_y_length += mv_y;
    sf->mv_x_sum_sqr += SQR(mv_x);
    sf->mv_y_sum_sqr += SQR(mv_y);
    sf->mv_length_diff += mvd_len;
    sf->mv_diff_sum_sqr += SQR(mvd_len);
#endif
}

static av_always_inline void clamp_mv(VP9mv *dst, const VP9mv *src,
                                      VP9TileData *td)
{
    dst->x = av_clip(src->x, td->min_mv.x, td->max_mv.x);
    dst->y = av_clip(src->y, td->min_mv.y, td->max_mv.y);
}

static void find_ref_mvs(VP9TileData *td,
                         VP9mv *pmv, int ref, int z, int idx, int sb)
{
    static const int8_t mv_ref_blk_off[N_BS_SIZES][8][2] = {
        [BS_64x64] = { {  3, -1 }, { -1,  3 }, {  4, -1 }, { -1,  4 },
                       { -1, -1 }, {  0, -1 }, { -1,  0 }, {  6, -1 } },
        [BS_64x32] = { {  0, -1 }, { -1,  0 }, {  4, -1 }, { -1,  2 },
                       { -1, -1 }, {  0, -3 }, { -3,  0 }, {  2, -1 } },
        [BS_32x64] = { { -1,  0 }, {  0, -1 }, { -1,  4 }, {  2, -1 },
                       { -1, -1 }, { -3,  0 }, {  0, -3 }, { -1,  2 } },
        [BS_32x32] = { {  1, -1 }, { -1,  1 }, {  2, -1 }, { -1,  2 },
                       { -1, -1 }, {  0, -3 }, { -3,  0 }, { -3, -3 } },
        [BS_32x16] = { {  0, -1 }, { -1,  0 }, {  2, -1 }, { -1, -1 },
                       { -1,  1 }, {  0, -3 }, { -3,  0 }, { -3, -3 } },
        [BS_16x32] = { { -1,  0 }, {  0, -1 }, { -1,  2 }, { -1, -1 },
                       {  1, -1 }, { -3,  0 }, {  0, -3 }, { -3, -3 } },
        [BS_16x16] = { {  0, -1 }, { -1,  0 }, {  1, -1 }, { -1,  1 },
                       { -1, -1 }, {  0, -3 }, { -3,  0 }, { -3, -3 } },
        [BS_16x8]  = { {  0, -1 }, { -1,  0 }, {  1, -1 }, { -1, -1 },
                       {  0, -2 }, { -2,  0 }, { -2, -1 }, { -1, -2 } },
        [BS_8x16]  = { { -1,  0 }, {  0, -1 }, { -1,  1 }, { -1, -1 },
                       { -2,  0 }, {  0, -2 }, { -1, -2 }, { -2, -1 } },
        [BS_8x8]   = { {  0, -1 }, { -1,  0 }, { -1, -1 }, {  0, -2 },
                       { -2,  0 }, { -1, -2 }, { -2, -1 }, { -2, -2 } },
        [BS_8x4]   = { {  0, -1 }, { -1,  0 }, { -1, -1 }, {  0, -2 },
                       { -2,  0 }, { -1, -2 }, { -2, -1 }, { -2, -2 } },
        [BS_4x8]   = { {  0, -1 }, { -1,  0 }, { -1, -1 }, {  0, -2 },
                       { -2,  0 }, { -1, -2 }, { -2, -1 }, { -2, -2 } },
        [BS_4x4]   = { {  0, -1 }, { -1,  0 }, { -1, -1 }, {  0, -2 },
                       { -2,  0 }, { -1, -2 }, { -2, -1 }, { -2, -2 } },
    };
    const VP9Context *s = td->s;
    VP9Block *b = td->b;
    int row = td->row, col = td->col, row7 = td->row7;
    const int8_t (*p)[2] = mv_ref_blk_off[b->bs];
#define INVALID_MV 0x80008000U
    uint32_t mem = INVALID_MV, mem_sub8x8 = INVALID_MV;
    int i;

#define RETURN_DIRECT_MV(mv)                    \
    do {                                        \
        uint32_t m = AV_RN32A(&mv);             \
        if (!idx) {                             \
            AV_WN32A(pmv, m);                   \
            return;                             \
        } else if (mem == INVALID_MV) {         \
            mem = m;                            \
        } else if (m != mem) {                  \
            AV_WN32A(pmv, m);                   \
            return;                             \
        }                                       \
    } while (0)

    if (sb >= 0) {
        if (sb == 2 || sb == 1) {
            RETURN_DIRECT_MV(b->mv[0][z]);
        } else if (sb == 3) {
            RETURN_DIRECT_MV(b->mv[2][z]);
            RETURN_DIRECT_MV(b->mv[1][z]);
            RETURN_DIRECT_MV(b->mv[0][z]);
        }

#define RETURN_MV(mv)                                                  \
    do {                                                               \
        if (sb > 0) {                                                  \
            VP9mv tmp;                                                 \
            uint32_t m;                                                \
            av_assert2(idx == 1);                                      \
            av_assert2(mem != INVALID_MV);                             \
            if (mem_sub8x8 == INVALID_MV) {                            \
                clamp_mv(&tmp, &mv, td);                               \
                m = AV_RN32A(&tmp);                                    \
                if (m != mem) {                                        \
                    AV_WN32A(pmv, m);                                  \
                    return;                                            \
                }                                                      \
                mem_sub8x8 = AV_RN32A(&mv);                            \
            } else if (mem_sub8x8 != AV_RN32A(&mv)) {                  \
                clamp_mv(&tmp, &mv, td);                               \
                m = AV_RN32A(&tmp);                                    \
                if (m != mem) {                                        \
                    AV_WN32A(pmv, m);                                  \
                } else {                                               \
                    /* BUG I'm pretty sure this isn't the intention */ \
                    AV_WN32A(pmv, 0);                                  \
                }                                                      \
                return;                                                \
            }                                                          \
        } else {                                                       \
            uint32_t m = AV_RN32A(&mv);                                \
            if (!idx) {                                                \
                clamp_mv(pmv, &mv, td);                                \
                return;                                                \
            } else if (mem == INVALID_MV) {                            \
                mem = m;                                               \
            } else if (m != mem) {                                     \
                clamp_mv(pmv, &mv, td);                                \
                return;                                                \
            }                                                          \
        }                                                              \
    } while (0)

        if (row > 0) {
            VP9mvrefPair *mv = &s->s.frames[CUR_FRAME].mv[(row - 1) * s->sb_cols * 8 + col];
            if (mv->ref[0] == ref)
                RETURN_MV(s->above_mv_ctx[2 * col + (sb & 1)][0]);
            else if (mv->ref[1] == ref)
                RETURN_MV(s->above_mv_ctx[2 * col + (sb & 1)][1]);
        }
        if (col > td->tile_col_start) {
            VP9mvrefPair *mv = &s->s.frames[CUR_FRAME].mv[row * s->sb_cols * 8 + col - 1];
            if (mv->ref[0] == ref)
                RETURN_MV(td->left_mv_ctx[2 * row7 + (sb >> 1)][0]);
            else if (mv->ref[1] == ref)
                RETURN_MV(td->left_mv_ctx[2 * row7 + (sb >> 1)][1]);
        }
        i = 2;
    } else {
        i = 0;
    }

    // previously coded MVs in this neighborhood, using same reference frame
    for (; i < 8; i++) {
        int c = p[i][0] + col, r = p[i][1] + row;

        if (c >= td->tile_col_start && c < s->cols &&
            r >= 0 && r < s->rows) {
            VP9mvrefPair *mv = &s->s.frames[CUR_FRAME].mv[r * s->sb_cols * 8 + c];

            if (mv->ref[0] == ref)
                RETURN_MV(mv->mv[0]);
            else if (mv->ref[1] == ref)
                RETURN_MV(mv->mv[1]);
        }
    }

    // MV at this position in previous frame, using same reference frame
    if (s->s.h.use_last_frame_mvs) {
        VP9mvrefPair *mv = &s->s.frames[REF_FRAME_MVPAIR].mv[row * s->sb_cols * 8 + col];

        if (!s->s.frames[REF_FRAME_MVPAIR].uses_2pass)
            ff_progress_frame_await(&s->s.frames[REF_FRAME_MVPAIR].tf, row >> 3);
        if (mv->ref[0] == ref)
            RETURN_MV(mv->mv[0]);
        else if (mv->ref[1] == ref)
            RETURN_MV(mv->mv[1]);
    }

#define RETURN_SCALE_MV(mv, scale)              \
    do {                                        \
        if (scale) {                            \
            VP9mv mv_temp = { -mv.x, -mv.y };   \
            RETURN_MV(mv_temp);                 \
        } else {                                \
            RETURN_MV(mv);                      \
        }                                       \
    } while (0)

    // previously coded MVs in this neighborhood, using different reference frame
    for (i = 0; i < 8; i++) {
        int c = p[i][0] + col, r = p[i][1] + row;

        if (c >= td->tile_col_start && c < s->cols && r >= 0 && r < s->rows) {
            VP9mvrefPair *mv = &s->s.frames[CUR_FRAME].mv[r * s->sb_cols * 8 + c];

            if (mv->ref[0] != ref && mv->ref[0] >= 0)
                RETURN_SCALE_MV(mv->mv[0],
                                s->s.h.signbias[mv->ref[0]] != s->s.h.signbias[ref]);
            if (mv->ref[1] != ref && mv->ref[1] >= 0 &&
                // BUG - libvpx has this condition regardless of whether
                // we used the first ref MV and pre-scaling
                AV_RN32A(&mv->mv[0]) != AV_RN32A(&mv->mv[1])) {
                RETURN_SCALE_MV(mv->mv[1], s->s.h.signbias[mv->ref[1]] != s->s.h.signbias[ref]);
            }
        }
    }

    // MV at this position in previous frame, using different reference frame
    if (s->s.h.use_last_frame_mvs) {
        VP9mvrefPair *mv = &s->s.frames[REF_FRAME_MVPAIR].mv[row * s->sb_cols * 8 + col];

        // no need to await_progress, because we already did that above
        if (mv->ref[0] != ref && mv->ref[0] >= 0)
            RETURN_SCALE_MV(mv->mv[0], s->s.h.signbias[mv->ref[0]] != s->s.h.signbias[ref]);
        if (mv->ref[1] != ref && mv->ref[1] >= 0 &&
            // BUG - libvpx has this condition regardless of whether
            // we used the first ref MV and pre-scaling
            AV_RN32A(&mv->mv[0]) != AV_RN32A(&mv->mv[1])) {
            RETURN_SCALE_MV(mv->mv[1], s->s.h.signbias[mv->ref[1]] != s->s.h.signbias[ref]);
        }
    }

    AV_ZERO32(pmv);
    clamp_mv(pmv, pmv, td);
#undef INVALID_MV
#undef RETURN_MV
#undef RETURN_SCALE_MV
}

static av_always_inline int read_mv_component(VP9TileData *td, int idx, int hp)
{
    const VP9Context *s = td->s;
    int bit, sign = vpx_rac_get_prob(td->c, s->prob.p.mv_comp[idx].sign);
    int n, c = vp89_rac_get_tree(td->c, ff_vp9_mv_class_tree,
                                 s->prob.p.mv_comp[idx].classes);

    td->counts.mv_comp[idx].sign[sign]++;
    td->counts.mv_comp[idx].classes[c]++;
    if (c) {
        int m;

        for (n = 0, m = 0; m < c; m++) {
            bit = vpx_rac_get_prob(td->c, s->prob.p.mv_comp[idx].bits[m]);
            n |= bit << m;
            td->counts.mv_comp[idx].bits[m][bit]++;
        }
        n <<= 3;
        bit = vp89_rac_get_tree(td->c, ff_vp9_mv_fp_tree,
                                s->prob.p.mv_comp[idx].fp);
        n  |= bit << 1;
        td->counts.mv_comp[idx].fp[bit]++;
        if (hp) {
            bit = vpx_rac_get_prob(td->c, s->prob.p.mv_comp[idx].hp);
            td->counts.mv_comp[idx].hp[bit]++;
            n |= bit;
        } else {
            n |= 1;
            // bug in libvpx - we count for bw entropy purposes even if the
            // bit wasn't coded
            td->counts.mv_comp[idx].hp[1]++;
        }
        n += 8 << c;
    } else {
        n = vpx_rac_get_prob(td->c, s->prob.p.mv_comp[idx].class0);
        td->counts.mv_comp[idx].class0[n]++;
        bit = vp89_rac_get_tree(td->c, ff_vp9_mv_fp_tree,
                                s->prob.p.mv_comp[idx].class0_fp[n]);
        td->counts.mv_comp[idx].class0_fp[n][bit]++;
        n = (n << 3) | (bit << 1);
        if (hp) {
            bit = vpx_rac_get_prob(td->c, s->prob.p.mv_comp[idx].class0_hp);
            td->counts.mv_comp[idx].class0_hp[bit]++;
            n |= bit;
        } else {
            n |= 1;
            // bug in libvpx - we count for bw entropy purposes even if the
            // bit wasn't coded
            td->counts.mv_comp[idx].class0_hp[1]++;
        }
    }

    return sign ? -(n + 1) : (n + 1);
}

void ff_vp9_fill_mv(VP9TileData *td, VP9mv *mv, int mode, int sb)
{
    const VP9Context *s = td->s;
    VP9Block *b = td->b;
    // videoparser
    SharedFrameInfo *sf = videoparser_get_shared_frame_info(s->s.frames[CUR_FRAME].tf.f);
    int mvd_x = 0, mvd_y = 0;  // videoparser: track coded MVD
    int is_newmv = 0;          // videoparser: whether this is NEWMV mode
    double frm_dist = 1.0;     // videoparser: frame distance for legacy normalization
    int coded_mv_cnt = 0;      // videoparser: count of non-zero MV joints (for legacy CodedMv)

#if VP_MV_POC_NORMALIZATION
    // videoparser: Calculate frame distance from PTS (legacy mode)
    // FrmDist = max(1, (current_PTS - ref_PTS) / duration)
    // This normalizes motion vectors by temporal distance to reference frame
    if (mode != ZEROMV && !s->s.h.keyframe && !s->s.h.intraonly && !b->intra) {
        AVFrame *cur_frame = s->s.frames[CUR_FRAME].tf.f;
        int64_t frame_duration = cur_frame->duration;

        if (frame_duration > 0) {
            // Get reference frame PTS using the block's reference
            // LEGACY BUG REPLICATION: The legacy VideoStatVP9.c used b->ref[0] DIRECTLY
            // as an index into refs[], NOT mapping through refidx[] like the decoder does.
            // This was a bug (mc code uses s->s.refs[s->s.h.refidx[b->ref[0]]]), but we
            // replicate it for compatibility with legacy output.
            // Legacy code: s->s.refs[ b->ref[0] ].f->pts (BUG: should use refidx mapping)
            int ref_buf_idx = b->ref[0];  // Legacy bug: use ref type directly as buffer index
            const ProgressFrame *ref_pf = &s->s.refs[ref_buf_idx];
            if (ref_pf->f) {
                int64_t cur_pts = cur_frame->pts;
                int64_t ref_pts = ref_pf->f->pts;
                double pts_diff = (double)(cur_pts - ref_pts);
                frm_dist = pts_diff / (double)frame_duration;
                if (frm_dist < 1.0) {
                    frm_dist = 1.0;
                }
#if VP_MV_DEBUG_FRMDIST
                // Debug: track average frm_dist per frame
                if (cur_pts != vp9_debug_last_pts) {
                    // New frame - print summary of previous frame
                    if (vp9_debug_frame_count > 0 && vp9_debug_frame_count <= 30 && vp9_debug_frmdist_count > 0) {
                        double avg_frmdist = vp9_debug_frmdist_sum / vp9_debug_frmdist_count;
                        fprintf(stderr, "[VP9 FrmDist DEBUG] frame=%d avg_frm_dist=%.4f (from %d blocks) refs: LAST=%d GOLD=%d ALT=%d\n",
                                vp9_debug_frame_count, avg_frmdist, vp9_debug_frmdist_count,
                                vp9_debug_ref_count[0], vp9_debug_ref_count[1], vp9_debug_ref_count[2]);
                    }
                    // Reset for new frame
                    vp9_debug_last_pts = cur_pts;
                    vp9_debug_frame_count++;
                    vp9_debug_frmdist_sum = 0.0;
                    vp9_debug_frmdist_count = 0;
                    vp9_debug_checked_init = 0;  // Reset init check flag
                    vp9_debug_ref_count[0] = vp9_debug_ref_count[1] = vp9_debug_ref_count[2] = 0;
                    // Print ref frame PTS for this new frame
                    if (vp9_debug_frame_count <= 10) {
                        int ri0 = s->s.h.refidx[0], ri1 = s->s.h.refidx[1], ri2 = s->s.h.refidx[2];
                        fprintf(stderr, "[VP9 Refs DEBUG] frame=%d cur_pts=%"PRId64"\n",
                                vp9_debug_frame_count, cur_pts);
                        fprintf(stderr, "  CORRECT (via refidx): LAST[%d]=%"PRId64" GOLD[%d]=%"PRId64" ALT[%d]=%"PRId64"\n",
                                ri0, s->s.refs[ri0].f ? s->s.refs[ri0].f->pts : -1,
                                ri1, s->s.refs[ri1].f ? s->s.refs[ri1].f->pts : -1,
                                ri2, s->s.refs[ri2].f ? s->s.refs[ri2].f->pts : -1);
                        fprintf(stderr, "  LEGACY (direct idx):  refs[0]=%"PRId64" refs[1]=%"PRId64" refs[2]=%"PRId64"\n",
                                s->s.refs[0].f ? s->s.refs[0].f->pts : -1,
                                s->s.refs[1].f ? s->s.refs[1].f->pts : -1,
                                s->s.refs[2].f ? s->s.refs[2].f->pts : -1);
                    }
                }
                // Check SharedFrameInfo initial state (once per frame)
                if (!vp9_debug_checked_init && vp9_debug_frame_count <= 10) {
                    vp9_debug_checked_init = 1;
                    fprintf(stderr, "[VP9 Init DEBUG] frame=%d mv_length=%.2f mb_mv_count=%d (SHOULD BE 0!)\n",
                            vp9_debug_frame_count, sf->mv_length, sf->mb_mv_count);
                }
                // Accumulate frm_dist for this block and track reference buffer (legacy bug: used as ref type)
                vp9_debug_frmdist_sum += frm_dist;
                vp9_debug_frmdist_count++;
                if (ref_buf_idx >= 0 && ref_buf_idx < 3) {
                    vp9_debug_ref_count[ref_buf_idx]++;
                }
#endif
            }
        }
        // If duration is 0, frm_dist stays at 1.0 (fallback)
    }
#endif

    if (mode == ZEROMV) {
        AV_ZERO32(&mv[0]);
        AV_ZERO32(&mv[1]);
#if VP_MV_POC_NORMALIZATION
        // videoparser: ZEROMV blocks still count as inter blocks in legacy mode
        // They contribute to mb_mv_count but have zero motion values
        // Note: skip blocks should NOT be counted, but at this point we don't
        // have reliable access to skip flag for ZEROMV. The legacy code handles
        // this differently by calling ModeStatistics after all MVs are filled.
        // For now, we count ZEROMV blocks; this may cause slight overcounting
        // if some ZEROMV blocks are also skip blocks.
        if (!b->skip) {
            int count = (sb == -1) ? vp9_bs_to_4x4_count[b->bs] : 1;
            sf->mb_mv_count += count;
            // ZEROMV has zero motion, so no MV accumulation needed
        }
#endif
    } else {
        int hp;
        int mvd_comp;  // videoparser: temporary for MVD component

        // FIXME cache this value and reuse for other subblocks
        find_ref_mvs(td, &mv[0], b->ref[0], 0, mode == NEARMV,
                     mode == NEWMV ? -1 : sb);
        // FIXME maybe move this code into find_ref_mvs()
        if ((mode == NEWMV || sb == -1) &&
            !(hp = s->s.h.highprecisionmvs &&
              abs(mv[0].x) < 64 && abs(mv[0].y) < 64)) {
            if (mv[0].y & 1) {
                if (mv[0].y < 0)
                    mv[0].y++;
                else
                    mv[0].y--;
            }
            if (mv[0].x & 1) {
                if (mv[0].x < 0)
                    mv[0].x++;
                else
                    mv[0].x--;
            }
        }
        if (mode == NEWMV) {
            enum MVJoint j;
            td->c->bit_count = 0; // videoparser: reset before MV decoding
            j = vp89_rac_get_tree(td->c, ff_vp9_mv_joint_tree,
                                               s->prob.p.mv_joint);

            td->counts.mv_joint[j]++;
            // videoparser: track non-zero MV joint for legacy CodedMv calculation
            // Legacy: b->CodedMv[idx] += (int)(j != MV_JOINT_ZERO)
            if (j != MV_JOINT_ZERO) {
                coded_mv_cnt++;
            }
            if (j >= MV_JOINT_V) {
                mvd_comp = read_mv_component(td, 0, hp);
                mv[0].y += mvd_comp;
                mvd_y += mvd_comp;  // videoparser: accumulate MVD
            }
            if (j & 1) {
                mvd_comp = read_mv_component(td, 1, hp);
                mv[0].x += mvd_comp;
                mvd_x += mvd_comp;  // videoparser: accumulate MVD
            }
            // videoparser: accumulate motion bits
            sf->motion_bit_count += td->c->bit_count;
#if !VP_MV_POC_NORMALIZATION
            // In legacy mode, mv_coded_count is handled in mv_statistics_vp9()
            // with outlier rejection and weighted counting
            sf->mv_coded_count++;
#endif
            is_newmv = 1;  // videoparser: mark as NEWMV mode
        }

        if (b->comp) {
            // FIXME cache this value and reuse for other subblocks
            find_ref_mvs(td, &mv[1], b->ref[1], 1, mode == NEARMV,
                         mode == NEWMV ? -1 : sb);
            if ((mode == NEWMV || sb == -1) &&
                !(hp = s->s.h.highprecisionmvs &&
                  abs(mv[1].x) < 64 && abs(mv[1].y) < 64)) {
                if (mv[1].y & 1) {
                    if (mv[1].y < 0)
                        mv[1].y++;
                    else
                        mv[1].y--;
                }
                if (mv[1].x & 1) {
                    if (mv[1].x < 0)
                        mv[1].x++;
                    else
                        mv[1].x--;
                }
            }
            if (mode == NEWMV) {
                enum MVJoint j;
                td->c->bit_count = 0; // videoparser: reset before MV decoding
                j = vp89_rac_get_tree(td->c, ff_vp9_mv_joint_tree,
                                                   s->prob.p.mv_joint);

                td->counts.mv_joint[j]++;
                // videoparser: track non-zero MV joint for legacy CodedMv calculation
                // Legacy: b->CodedMv[idx] += (int)(j != MV_JOINT_ZERO)
                if (j != MV_JOINT_ZERO) {
                    coded_mv_cnt++;
                }
                if (j >= MV_JOINT_V) {
                    mvd_comp = read_mv_component(td, 0, hp);
                    mv[1].y += mvd_comp;
                    mvd_y += mvd_comp;  // videoparser: accumulate MVD
                }
                if (j & 1) {
                    mvd_comp = read_mv_component(td, 1, hp);
                    mv[1].x += mvd_comp;
                    mvd_x += mvd_comp;  // videoparser: accumulate MVD
                }
                // videoparser: accumulate motion bits
                sf->motion_bit_count += td->c->bit_count;
#if !VP_MV_POC_NORMALIZATION
                // In legacy mode, mv_coded_count is handled in mv_statistics_vp9()
                // with outlier rejection and weighted counting
                sf->mv_coded_count++;
#endif
            }
        }

        // videoparser: accumulate MV statistics for inter blocks
        // Pass block size, sub-block index, skip flag, and coded_mv_cnt for proper counting in legacy mode
        mv_statistics_vp9(sf, mv, b->comp, mvd_x, mvd_y, is_newmv, frm_dist, b->bs, sb, b->skip, coded_mv_cnt);
    }
}
