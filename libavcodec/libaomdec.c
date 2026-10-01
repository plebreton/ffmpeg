/*
 * Copyright (c) 2010, Google, Inc.
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

/**
 * @file
 * AV1 decoder support via libaom
 */

#include <aom/aom_decoder.h>
#include <aom/aomdx.h>

#include "libavutil/common.h"
#include "libavutil/cpu.h"
#include "libavutil/hdr_dynamic_metadata.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"

#include "avcodec.h"
#include "bytestream.h"
#include "codec_internal.h"
#include "decode.h"
#include "itut35.h"
#include "libaom.h"
#include "profiles.h"
#include "videoparser_export.h"

// videoparser: Include inspection API for MV extraction
#ifdef AOM_CTRL_AV1_SET_INSPECTION_CALLBACK
#include "av1/decoder/inspection.h"
#include "av1/common/enums.h"
#include "av1/common/common_data.h"
#include <math.h>
#endif

// videoparser: export data of the last decoded frames, until they are output
#define VP_EXPORT_CACHE_SIZE 16

typedef struct VPExportCacheEntry {
    const void *key;      ///< libaom frame buffer of the frame
    AVBufferRef *buf;     ///< VPExportHeader and data
} VPExportCacheEntry;

// videoparser: Helper macro
#define VP_SQR(_x_) ((_x_) * (_x_))

typedef struct AV1DecodeContext {
    const AVClass *class;
    struct aom_codec_ctx decoder;
    // videoparser: per-block QP, MV and bits exports
    char *qp_export_path;
    char *mv_export_path;
    char *bits_export_path;
    VPExportFiles vp_export_files;
    VPExportCacheEntry vp_export_cache[VP_EXPORT_CACHE_SIZE];
    int vp_export_cache_next;
    int32_t vp_export_counter;  ///< number of output frames written
    int vp_export_error;        ///< error from the inspection callback
#ifdef AOM_CTRL_AV1_SET_INSPECTION_CALLBACK
    // videoparser: Inspection data for MV extraction
    insp_frame_data insp_data;
    int insp_data_initialized;
    int insp_data_valid;  // Set when inspection callback has been called
#endif
} AV1DecodeContext;

#ifdef AOM_CTRL_AV1_SET_INSPECTION_CALLBACK
/**
 * videoparser: spread the bits of a block (in 1/8 bits) over its 4x4 units in
 * the frame. Each value is converted to whole bits from its running sum over
 * the frame, so that the sum over the frame is the frame's count in whole bits
 * (as motion_bit_count and coefs_bit_count). The other bits are spread
 * separately, so that motion + coeff <= total holds in every unit.
 */
static void videoparser_av1_spread_bits(VPExportBits *bits, int stride,
                                        int row, int col, int w, int h,
                                        const uint32_t value[3], uint64_t sum[3])
{
    const int n = w * h;
    int k = 0;

    for (int y = row; y < row + h; y++) {
        for (int x = col; x < col + w; x++, k++) {
            VPExportBits *b = &bits[y * stride + x];
            uint32_t out[3];

            for (int i = 0; i < 3; i++) {
                const uint32_t share = value[i] / n + (k < value[i] % n);
                out[i] = (uint32_t)(((sum[i] + share) >> 3) - (sum[i] >> 3));
                sum[i] += share;
            }
            b->motion_bits += out[0];
            b->coeff_bits  += out[1];
            b->total_bits  += out[0] + out[1] + out[2];
        }
    }
}

/**
 * videoparser: store the QP, MV and bits of the decoded frame for the
 * exports, until the frame is output
 */
static void videoparser_av1_store_export(AV1DecodeContext *ctx)
{
    const insp_frame_data *fd = &ctx->insp_data;
    const int rows = fd->mi_rows, cols = fd->mi_cols;
    VPExportCacheEntry *entry = NULL;
    VPExportHeader *h;
    AVBufferRef *buf;
    uint64_t sum[3] = { 0 };
    int ret;

    buf = vp_export_alloc_buffer(&ctx->vp_export_files, 0, cols, rows, cols, rows,
                                 cols, rows, 4, &ret);
    if (!buf) {
        if (ret < 0)
            ctx->vp_export_error = ret;
        return;
    }
    h = (VPExportHeader *)buf->data;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            const insp_mi_data *mi = &fd->mi_grid[y * cols + x];

            if (h->qp_w)
                vp_export_qp(h)[y * cols + x] = mi->vp_qindex;

            if (h->mv_w && mi->ref_frame[0] > INTRA_FRAME) {
                VPExportMV *mv = &vp_export_mv(h)[y * cols + x];
                mv->mv_l0_x    = mi->mv[0].col;
                mv->mv_l0_y    = mi->mv[0].row;
                mv->ref_idx_l0 = mi->ref_frame[0];
                mv->pred_flag  = 1;
                if (mi->ref_frame[1] > INTRA_FRAME) {
                    mv->mv_l1_x    = mi->mv[1].col;
                    mv->mv_l1_y    = mi->mv[1].row;
                    mv->ref_idx_l1 = mi->ref_frame[1];
                    mv->pred_flag |= 2;
                }
            }

            // the bits are stored in each 4x4 unit of a block; spread them
            // from the top-left unit of the block
            if (h->bits_w && mi->vp_mi_row == y && mi->vp_mi_col == x &&
                mi->bsize >= 0 && mi->bsize < BLOCK_SIZES_ALL) {
                const uint32_t total = mi->vp_total_bits;
                const uint32_t motion = FFMIN(mi->vp_motion_bits, total);
                const uint32_t coeff = FFMIN(mi->vp_coeff_bits, total - motion);
                const uint32_t value[3] = { motion, coeff, total - motion - coeff };
                const int w = FFMIN(mi_size_wide[mi->bsize], cols - x);
                const int bh = FFMIN(mi_size_high[mi->bsize], rows - y);

                videoparser_av1_spread_bits(vp_export_bits(h), cols, y, x, w, bh,
                                            value, sum);
            }
        }
    }

    // replace the data of a frame buffer that is decoded again, or the oldest
    for (int i = 0; i < VP_EXPORT_CACHE_SIZE; i++) {
        if (ctx->vp_export_cache[i].buf && ctx->vp_export_cache[i].key == fd->vp_frame_key) {
            entry = &ctx->vp_export_cache[i];
            break;
        }
    }
    if (!entry) {
        entry = &ctx->vp_export_cache[ctx->vp_export_cache_next];
        ctx->vp_export_cache_next = (ctx->vp_export_cache_next + 1) % VP_EXPORT_CACHE_SIZE;
    }
    av_buffer_unref(&entry->buf);
    entry->key = fd->vp_frame_key;
    entry->buf = buf;
}

/**
 * videoparser: write the exports of an output frame, numbered in output order
 */
static int videoparser_av1_write_export(AV1DecodeContext *ctx, const struct aom_image *img)
{
    VPExportHeader *h = NULL;

    if (ctx->vp_export_error < 0)
        return ctx->vp_export_error;

    for (int i = 0; i < VP_EXPORT_CACHE_SIZE; i++) {
        if (ctx->vp_export_cache[i].buf && ctx->vp_export_cache[i].key == img->vp_frame_key) {
            h = (VPExportHeader *)ctx->vp_export_cache[i].buf->data;
            break;
        }
    }
    if (h)
        h->id = ctx->vp_export_counter;
    ctx->vp_export_counter++;
    return vp_export_write_header(&ctx->vp_export_files, h);
}

// videoparser: Inspection callback to capture frame data for MV extraction
// The callback signature from libaom is: void (*aom_inspect_cb)(void *decoder, void *ctx)
static void videoparser_av1_inspect_callback(void *pbi, void *user_data) {
    AV1DecodeContext *ctx = (AV1DecodeContext *)user_data;
    if (!ctx)
        return;

    ctx->insp_data_valid = 0;

    // Call ifd_inspect to fill the inspection data
    // Note: This captures MV, mode, and other per-block information
    if (ifd_inspect(&ctx->insp_data, pbi, 0) == 1) {
        ctx->insp_data_valid = 1;
        videoparser_av1_store_export(ctx);
    }
}
#endif

static av_cold int aom_init(AVCodecContext *avctx,
                            const struct aom_codec_iface *iface)
{
    AV1DecodeContext *ctx           = avctx->priv_data;
    struct aom_codec_dec_cfg deccfg = {
        .threads = FFMIN(avctx->thread_count ? avctx->thread_count : av_cpu_count(), 16)
    };

    av_log(avctx, AV_LOG_VERBOSE, "%s\n", aom_codec_version_str());
    av_log(avctx, AV_LOG_VERBOSE, "%s\n", aom_codec_build_config());

    // videoparser
    {
        int ret = vp_export_open(&ctx->vp_export_files, ctx->qp_export_path,
                                 ctx->mv_export_path, ctx->bits_export_path);
        if (ret < 0)
            return ret;
    }

    if (aom_codec_dec_init(&ctx->decoder, iface, &deccfg, 0) != AOM_CODEC_OK) {
        const char *error = aom_codec_error(&ctx->decoder);
        av_log(avctx, AV_LOG_ERROR, "Failed to initialize decoder: %s\n",
               error);
        return AVERROR(EINVAL);
    }

#ifdef AOM_CTRL_AV1_SET_INSPECTION_CALLBACK
    // videoparser: Set up inspection callback for MV extraction
    {
        aom_inspect_init ii;
        ii.inspect_cb = videoparser_av1_inspect_callback;
        ii.inspect_ctx = ctx;
        ctx->insp_data_initialized = 0;
        ctx->insp_data_valid = 0;
        ctx->insp_data.mi_grid = NULL;

        if (aom_codec_control(&ctx->decoder, AV1_SET_INSPECTION_CALLBACK, &ii) != AOM_CODEC_OK) {
            av_log(avctx, AV_LOG_WARNING, "Failed to set inspection callback, MV extraction disabled\n");
        } else {
            av_log(avctx, AV_LOG_VERBOSE, "videoparser: AV1 inspection callback enabled for MV extraction\n");
        }
    }
#endif

    return 0;
}

// returns 0 on success, AVERROR_INVALIDDATA otherwise
static int set_pix_fmt(AVCodecContext *avctx, struct aom_image *img)
{
    static const enum AVColorRange color_ranges[] = {
        AVCOL_RANGE_MPEG, AVCOL_RANGE_JPEG
    };
    avctx->color_range = color_ranges[img->range];
    if (img->cp != AOM_CICP_CP_UNSPECIFIED)
        avctx->color_primaries = (enum AVColorPrimaries)img->cp;
    if (img->mc != AOM_CICP_MC_UNSPECIFIED)
        avctx->colorspace  = (enum AVColorSpace)img->mc;
    if (img->tc != AOM_CICP_TC_UNSPECIFIED)
        avctx->color_trc   = (enum AVColorTransferCharacteristic)img->tc;

    switch (img->fmt) {
    case AOM_IMG_FMT_I420:
    case AOM_IMG_FMT_I42016:
        if (img->bit_depth == 8) {
            avctx->pix_fmt = img->monochrome ?
                             AV_PIX_FMT_GRAY8 : AV_PIX_FMT_YUV420P;
            avctx->profile = AV_PROFILE_AV1_MAIN;
            return 0;
        } else if (img->bit_depth == 10) {
            avctx->pix_fmt = img->monochrome ?
                             AV_PIX_FMT_GRAY10 : AV_PIX_FMT_YUV420P10;
            avctx->profile = AV_PROFILE_AV1_MAIN;
            return 0;
        } else if (img->bit_depth == 12) {
            avctx->pix_fmt = img->monochrome ?
                             AV_PIX_FMT_GRAY12 : AV_PIX_FMT_YUV420P12;
            avctx->profile = AV_PROFILE_AV1_PROFESSIONAL;
            return 0;
        } else {
            return AVERROR_INVALIDDATA;
        }
    case AOM_IMG_FMT_I422:
    case AOM_IMG_FMT_I42216:
        if (img->bit_depth == 8) {
            avctx->pix_fmt = AV_PIX_FMT_YUV422P;
            avctx->profile = AV_PROFILE_AV1_PROFESSIONAL;
            return 0;
        } else if (img->bit_depth == 10) {
            avctx->pix_fmt = AV_PIX_FMT_YUV422P10;
            avctx->profile = AV_PROFILE_AV1_PROFESSIONAL;
            return 0;
        } else if (img->bit_depth == 12) {
            avctx->pix_fmt = AV_PIX_FMT_YUV422P12;
            avctx->profile = AV_PROFILE_AV1_PROFESSIONAL;
            return 0;
        } else {
            return AVERROR_INVALIDDATA;
        }
    case AOM_IMG_FMT_I444:
    case AOM_IMG_FMT_I44416:
        if (img->bit_depth == 8) {
            avctx->pix_fmt = avctx->colorspace == AVCOL_SPC_RGB ?
                             AV_PIX_FMT_GBRP : AV_PIX_FMT_YUV444P;
            avctx->profile = AV_PROFILE_AV1_HIGH;
            return 0;
        } else if (img->bit_depth == 10) {
            avctx->pix_fmt = AV_PIX_FMT_YUV444P10;
            avctx->pix_fmt = avctx->colorspace == AVCOL_SPC_RGB ?
                             AV_PIX_FMT_GBRP10 : AV_PIX_FMT_YUV444P10;
            avctx->profile = AV_PROFILE_AV1_HIGH;
            return 0;
        } else if (img->bit_depth == 12) {
            avctx->pix_fmt = avctx->colorspace == AVCOL_SPC_RGB ?
                             AV_PIX_FMT_GBRP12 : AV_PIX_FMT_YUV444P12;
            avctx->profile = AV_PROFILE_AV1_PROFESSIONAL;
            return 0;
        } else {
            return AVERROR_INVALIDDATA;
        }

    default:
        return AVERROR_INVALIDDATA;
    }
}

static int decode_metadata_itu_t_t35(AVCodecContext *avctx, AVFrame *frame,
                                     const uint8_t *buffer, size_t buffer_size)
{
    FFITUTT35 itut35 = { 0 };
    int ret;

    if (buffer_size < 6)
        return AVERROR(EINVAL);

    ret = ff_itut_t35_parse_buffer(&itut35, buffer, buffer_size, 0);
    if (ret <= 0)
        return ret;

    ret = ff_itut_t35_parse_payload_to_frame(&itut35, NULL, avctx, frame);
    if (ret < 0)
        return ret;

    return 0;
}

static int decode_metadata(AVCodecContext *avctx, AVFrame *frame,
                           const struct aom_image *img)
{
    const size_t num_metadata = aom_img_num_metadata(img);
    for (size_t i = 0; i < num_metadata; ++i) {
        const aom_metadata_t *metadata = aom_img_get_metadata(img, i);
        if (!metadata)
            continue;

        switch (metadata->type) {
        case OBU_METADATA_TYPE_ITUT_T35: {
            int res = decode_metadata_itu_t_t35(avctx, frame, metadata->payload, metadata->sz);
            if (res < 0)
                return res;
            break;
        }
        default:
            break;
        }
    }
    return 0;
}

#ifdef AOM_CTRL_AV1_SET_INSPECTION_CALLBACK
/**
 * videoparser: Extract motion vector statistics from AV1 inspection data.
 * This is called after decoding each frame to populate SharedFrameInfo.
 *
 * AV1 motion vectors are in 1/8 pel units (same as VP9).
 * We iterate over all MI blocks and extract MV data for inter blocks.
 */
static void videoparser_av1_extract_mv_stats(AVFrame *picture, AV1DecodeContext *ctx)
{
    if (!ctx->insp_data_valid || !ctx->insp_data.mi_grid) {
        av_log(NULL, AV_LOG_DEBUG, "videoparser: AV1 MV extraction skipped - insp_data_valid=%d, mi_grid=%p\n",
               ctx->insp_data_valid, (void*)ctx->insp_data.mi_grid);
        return;
    }

    SharedFrameInfo *sf = videoparser_get_shared_frame_info(picture);
    if (!sf)
        return;

    const insp_frame_data *fd = &ctx->insp_data;
    const int mi_rows = fd->mi_rows;
    const int mi_cols = fd->mi_cols;
    int inter_blocks = 0;  // Debug counter

    // Iterate over all MI blocks
    for (int mi_row = 0; mi_row < mi_rows; mi_row++) {
        for (int mi_col = 0; mi_col < mi_cols; mi_col++) {
            const insp_mi_data *mi = &fd->mi_grid[mi_row * mi_cols + mi_col];

            // Check if this is an inter block (mode >= NEARESTMV)
            // In AV1/libaom enums: NEARESTMV=13, NEARMV=14, GLOBALMV=15, NEWMV=16,
            // and compound modes start from NEAREST_NEARESTMV=17
            // Intra modes are DC_PRED=0 to PAETH_PRED=12
            if (mi->mode < NEARESTMV)
                continue;  // Skip intra blocks

            // Skip blocks with INTRA_FRAME reference (ref_frame[0] == 0 means INTRA_FRAME)
            // In AV1: INTRA_FRAME=0, LAST_FRAME=1, etc.
            if (mi->ref_frame[0] <= 0)
                continue;

            double mv_x = 0.0, mv_y = 0.0;
            int dir_cnt = 0;

            // L0 reference
            if (mi->ref_frame[0] > 0) {
                dir_cnt++;
                mv_x += fabs((double)mi->mv[0].col);
                mv_y += fabs((double)mi->mv[0].row);
            }

            // L1 reference (compound mode)
            if (mi->ref_frame[1] > 0) {
                dir_cnt++;
                mv_x += fabs((double)mi->mv[1].col);
                mv_y += fabs((double)mi->mv[1].row);
            }

            if (dir_cnt == 0)
                continue;

            // Average across directions for compound blocks
            if (dir_cnt > 1) {
                mv_x /= dir_cnt;
                mv_y /= dir_cnt;
            }

            // Calculate magnitude
            double mv_length_xy = sqrt(VP_SQR(mv_x) + VP_SQR(mv_y));

            // videoparser: Extract MVD (motion vector difference) from inspection data
            // MVD is now captured during decoding in libaom
            double mvd_x = 0.0, mvd_y = 0.0;
            int mvd_dir_cnt = 0;

            // L0 MVD
            if (mi->ref_frame[0] > 0 && (mi->mvd[0].col != 0 || mi->mvd[0].row != 0)) {
                mvd_dir_cnt++;
                mvd_x += fabs((double)mi->mvd[0].col);
                mvd_y += fabs((double)mi->mvd[0].row);
            }

            // L1 MVD (compound mode)
            if (mi->ref_frame[1] > 0 && (mi->mvd[1].col != 0 || mi->mvd[1].row != 0)) {
                mvd_dir_cnt++;
                mvd_x += fabs((double)mi->mvd[1].col);
                mvd_y += fabs((double)mi->mvd[1].row);
            }

            // Average across directions for compound blocks
            if (mvd_dir_cnt > 1) {
                mvd_x /= mvd_dir_cnt;
                mvd_y /= mvd_dir_cnt;
            }

            double mvd_len = sqrt(VP_SQR(mvd_x) + VP_SQR(mvd_y));

            // Accumulate statistics
            sf->mv_length += mv_length_xy;
            sf->mv_sum_sqr += VP_SQR(mv_length_xy);
            sf->mv_x_length += mv_x;
            sf->mv_y_length += mv_y;
            sf->mv_x_sum_sqr += VP_SQR(mv_x);
            sf->mv_y_sum_sqr += VP_SQR(mv_y);
            sf->mv_length_diff += mvd_len;
            sf->mv_diff_sum_sqr += VP_SQR(mvd_len);

            sf->mb_mv_count++;

            // Count coded MVs (NEWMV and compound NEWMV modes)
            // NEWMV=16, NEW_NEWMV=24, NEAREST_NEWMV=19, NEW_NEARESTMV=20, etc.
            if (mi->mode == NEWMV || mi->mode == NEW_NEWMV ||
                mi->mode == NEAREST_NEWMV || mi->mode == NEW_NEARESTMV ||
                mi->mode == NEAR_NEWMV || mi->mode == NEW_NEARMV) {
                sf->mv_coded_count++;
            }
            inter_blocks++;
        }
    }

    // videoparser: Extract bit counts from inspection data
    sf->motion_bit_count = fd->motion_bits;
    sf->coefs_bit_count = fd->coef_bits;

    av_log(NULL, AV_LOG_DEBUG, "videoparser: AV1 frame has %d inter blocks out of %d total MI blocks, mb_mv_count=%d, motion_bits=%llu, coef_bits=%llu\n",
           inter_blocks, mi_rows * mi_cols, sf->mb_mv_count,
           (unsigned long long)sf->motion_bit_count, (unsigned long long)sf->coefs_bit_count);
}
#endif

static int aom_decode(AVCodecContext *avctx, AVFrame *picture,
                      int *got_frame, AVPacket *avpkt)
{
    AV1DecodeContext *ctx = avctx->priv_data;
    const void *iter      = NULL;
    struct aom_image *img;
    int ret;

    if (aom_codec_decode(&ctx->decoder, avpkt->data, avpkt->size, NULL) !=
        AOM_CODEC_OK) {
        const char *error  = aom_codec_error(&ctx->decoder);
        const char *detail = aom_codec_error_detail(&ctx->decoder);

        av_log(avctx, AV_LOG_ERROR, "Failed to decode frame: %s\n", error);
        if (detail)
            av_log(avctx, AV_LOG_ERROR, "  Additional information: %s\n",
                   detail);
        return AVERROR_INVALIDDATA;
    }

    // videoparser
    int qp;
    aom_codec_control(&ctx->decoder, AOMD_GET_LAST_QUANTIZER, &qp);
    videoparser_shared_frame_info_update_qp(picture, qp);

    if ((img = aom_codec_get_frame(&ctx->decoder, &iter))) {
        if (img->d_w > img->w || img->d_h > img->h) {
            av_log(avctx, AV_LOG_ERROR, "Display dimensions %dx%d exceed storage %dx%d\n",
                   img->d_w, img->d_h, img->w, img->h);
            return AVERROR_EXTERNAL;
        }

        if ((ret = set_pix_fmt(avctx, img)) < 0) {
            av_log(avctx, AV_LOG_ERROR, "Unsupported output colorspace (%d) / bit_depth (%d)\n",
                   img->fmt, img->bit_depth);
            return ret;
        }

        if ((int)img->d_w != avctx->width || (int)img->d_h != avctx->height) {
            av_log(avctx, AV_LOG_INFO, "dimension change! %dx%d -> %dx%d\n",
                   avctx->width, avctx->height, img->d_w, img->d_h);
            ret = ff_set_dimensions(avctx, img->d_w, img->d_h);
            if (ret < 0)
                return ret;
        }
        if ((ret = ff_get_buffer(avctx, picture, 0)) < 0)
            return ret;

#ifdef AOM_CTRL_AOMD_GET_FRAME_FLAGS
        {
            aom_codec_frame_flags_t flags;
            ret = aom_codec_control(&ctx->decoder, AOMD_GET_FRAME_FLAGS, &flags);
            if (ret == AOM_CODEC_OK) {
                if (flags & AOM_FRAME_IS_KEY)
                    picture->flags |= AV_FRAME_FLAG_KEY;
                else
                    picture->flags &= ~AV_FRAME_FLAG_KEY;
                if (flags & (AOM_FRAME_IS_KEY | AOM_FRAME_IS_INTRAONLY))
                    picture->pict_type = AV_PICTURE_TYPE_I;
                else if (flags & AOM_FRAME_IS_SWITCH)
                    picture->pict_type = AV_PICTURE_TYPE_SP;
                else
                    picture->pict_type = AV_PICTURE_TYPE_P;
            }
        }
#endif

        av_reduce(&picture->sample_aspect_ratio.num,
                  &picture->sample_aspect_ratio.den,
                  picture->height * img->r_w,
                  picture->width * img->r_h,
                  INT_MAX);
        ff_set_sar(avctx, picture->sample_aspect_ratio);

        if ((img->fmt & AOM_IMG_FMT_HIGHBITDEPTH) && img->bit_depth == 8)
            ff_aom_image_copy_16_to_8(picture, img);
        else {
            const uint8_t *planes[4] = { img->planes[0], img->planes[1], img->planes[2] };
            const int      stride[4] = { img->stride[0], img->stride[1], img->stride[2] };

            av_image_copy(picture->data, picture->linesize, planes,
                          stride, avctx->pix_fmt, img->d_w, img->d_h);
        }
        ret = decode_metadata(avctx, picture, img);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "Failed to decode metadata\n");
            return ret;
        }

#ifdef AOM_CTRL_AV1_SET_INSPECTION_CALLBACK
        // videoparser: Extract MV statistics from inspection data
        videoparser_av1_extract_mv_stats(picture, ctx);

        // videoparser: per-block exports of the output frame
        if ((ret = videoparser_av1_write_export(ctx, img)) < 0)
            return ret;
#endif

        *got_frame = 1;
    }
    return avpkt->size;
}

static av_cold int aom_free(AVCodecContext *avctx)
{
    AV1DecodeContext *ctx = avctx->priv_data;

#ifdef AOM_CTRL_AV1_SET_INSPECTION_CALLBACK
    // videoparser: Clean up inspection data
    if (ctx->insp_data.mi_grid) {
        ifd_clear(&ctx->insp_data);
    }
    for (int i = 0; i < VP_EXPORT_CACHE_SIZE; i++)
        av_buffer_unref(&ctx->vp_export_cache[i].buf);
#endif
    vp_export_close(&ctx->vp_export_files); // videoparser

    aom_codec_destroy(&ctx->decoder);
    return 0;
}

static av_cold int av1_init(AVCodecContext *avctx)
{
    return aom_init(avctx, aom_codec_av1_dx());
}

// videoparser: options for the per-block exports
#define OFFSET(x) offsetof(AV1DecodeContext, x)
#define VD AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM
static const AVOption options[] = {
    { "export_qp_matrix", "Write per-frame AV1 QP matrices to a binary file", OFFSET(qp_export_path), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, VD },
    { "export_mv_matrix", "Write per-frame AV1 motion-vector matrix to a binary file", OFFSET(mv_export_path), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, VD },
    { "export_ctu_bits_matrix", "Write per-frame AV1 block bit counts to a binary file", OFFSET(bits_export_path), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, VD },
    { NULL }
};

static const AVClass libaom_av1_class = {
    .class_name = "libaom-av1 decoder",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_libaom_av1_decoder = {
    .p.name         = "libaom-av1",
    CODEC_LONG_NAME("libaom AV1"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_AV1,
    .priv_data_size = sizeof(AV1DecodeContext),
    .p.priv_class   = &libaom_av1_class,
    .init           = av1_init,
    .close          = aom_free,
    FF_CODEC_DECODE_CB(aom_decode),
    .p.capabilities = AV_CODEC_CAP_OTHER_THREADS | AV_CODEC_CAP_DR1,
    .caps_internal  = FF_CODEC_CAP_NOT_INIT_THREADSAFE |
                      FF_CODEC_CAP_AUTO_THREADS,
    .p.profiles     = NULL_IF_CONFIG_SMALL(ff_av1_profiles),
    .p.wrapper_name = "libaom",
};
