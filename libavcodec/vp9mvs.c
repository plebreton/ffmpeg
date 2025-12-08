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

/**
 * Raw motion vector statistics extraction for VP9.
 * Extracts MV values without any normalization.
 * Motion vectors are in 1/8 pel units (VP9 uses 1/8 pel precision).
 *
 * @param sf SharedFrameInfo to accumulate statistics into
 * @param mv Motion vector array [0]=L0, [1]=L1
 * @param comp Whether this is compound (bi-predictive) mode
 * @param mvd_x MVD x component (only valid for NEWMV mode)
 * @param mvd_y MVD y component (only valid for NEWMV mode)
 * @param has_mvd Whether MVD values are valid
 */
static void mv_statistics_vp9(SharedFrameInfo *sf, const VP9mv *mv,
                              int comp, int mvd_x, int mvd_y, int has_mvd) {
    double mv_x = 0.0, mv_y = 0.0;
    double mvd_len = 0.0;
    double mv_length_xy;
    int dir_cnt = 0;

    // L0 reference (always present for inter blocks)
    dir_cnt++;
    mv_x = fabs((double)mv[0].x);
    mv_y = fabs((double)mv[0].y);

    // L1 reference (compound/bi-predictive)
    if (comp) {
        dir_cnt++;
        mv_x += fabs((double)mv[1].x);
        mv_y += fabs((double)mv[1].y);
    }

    // Average across directions for bi-predictive blocks
    if (dir_cnt > 1) {
        mv_x /= dir_cnt;
        mv_y /= dir_cnt;
    }

    // Calculate magnitude
    mv_length_xy = sqrt(SQR(mv_x) + SQR(mv_y));

    // For diff stats, use the coded MVD if available (NEWMV mode)
    if (has_mvd) {
        mvd_len = sqrt(SQR((double)mvd_x) + SQR((double)mvd_y));
    }

    // Accumulate statistics
    sf->mv_length += mv_length_xy;
    sf->mv_sum_sqr += SQR(mv_length_xy);
    sf->mv_x_length += mv_x;
    sf->mv_y_length += mv_y;
    sf->mv_x_sum_sqr += SQR(mv_x);
    sf->mv_y_sum_sqr += SQR(mv_y);
    sf->mv_length_diff += mvd_len;
    sf->mv_diff_sum_sqr += SQR(mvd_len);

    sf->mb_mv_count++;
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
    int has_mvd = 0;           // videoparser: whether MVD was coded

    if (mode == ZEROMV) {
        AV_ZERO32(&mv[0]);
        AV_ZERO32(&mv[1]);
        // ZEROMV has no motion to track
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
            sf->mv_coded_count++;
            has_mvd = 1;
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
                sf->mv_coded_count++;
            }
        }

        // videoparser: accumulate MV statistics for inter blocks
        mv_statistics_vp9(sf, mv, b->comp, mvd_x, mvd_y, has_mvd);
    }
}
