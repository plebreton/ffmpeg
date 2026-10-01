/*
 * videoparser: per-block QP, motion vector and bit exports
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
 * videoparser: per-block exports that travel with the decoded frame.
 *
 * The decoder attaches the export buffer to the frame as side data when it
 * starts decoding it, fills it while decoding, and writes it to the export
 * files when the frame is returned. The exports are therefore in output
 * order, and a frame that is shown again (VP9 show_existing_frame) carries
 * its own data.
 *
 * File formats (little endian), one record per output frame:
 * - QP:   int32 id, width, height; int16 QP per block (-1: skipped block)
 * - MV:   int32 id, width, height; VPExportMV per block
 * - bits: int32 id, width, height, block size; VPExportBits per block
 */

#ifndef AVCODEC_VIDEOPARSER_EXPORT_H
#define AVCODEC_VIDEOPARSER_EXPORT_H

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "libavutil/buffer.h"
#include "libavutil/error.h"
#include "libavutil/frame.h"

typedef struct VPExportMV {
    int16_t mv_l0_x;
    int16_t mv_l0_y;
    int16_t mv_l1_x;
    int16_t mv_l1_y;
    int8_t  ref_idx_l0;
    int8_t  ref_idx_l1;
    int8_t  pred_flag;  ///< 0: intra or not coded, 1: L0, 2: L1, 3: both
    int8_t  reserved;
} VPExportMV;

typedef struct VPExportBits {
    uint32_t total_bits;   ///< all bits of the block
    uint32_t motion_bits;  ///< bits of the motion information
    uint32_t coeff_bits;   ///< bits of the residual
} VPExportBits;

typedef struct VPExportHeader {
    int32_t id;            ///< written as the first header field (POC or counter)
    int32_t qp_w, qp_h;    ///< 0 if the QP export is off
    int32_t mv_w, mv_h;    ///< 0 if the MV export is off
    int32_t bits_w, bits_h, bits_block_size; ///< 0 if the bits export is off
    int32_t qp_offset, mv_offset, bits_offset; ///< from the start of the header
} VPExportHeader;

typedef struct VPExportFiles {
    FILE *qp;
    FILE *mv;
    FILE *bits;
} VPExportFiles;

static inline int16_t *vp_export_qp(VPExportHeader *h)
{
    return (int16_t *)((uint8_t *)h + h->qp_offset);
}

static inline VPExportMV *vp_export_mv(VPExportHeader *h)
{
    return (VPExportMV *)((uint8_t *)h + h->mv_offset);
}

static inline VPExportBits *vp_export_bits(VPExportHeader *h)
{
    return (VPExportBits *)((uint8_t *)h + h->bits_offset);
}

static inline void vp_export_mv_none(VPExportMV *mv)
{
    mv->mv_l0_x = mv->mv_l0_y = -32768;
    mv->mv_l1_x = mv->mv_l1_y = -32768;
    mv->ref_idx_l0 = -1;
    mv->ref_idx_l1 = -1;
    mv->pred_flag  = 0;
    mv->reserved   = 0;
}

/**
 * Allocate a zeroed export buffer for the exports whose file is open.
 * Returns NULL if no export is on, or on allocation failure (with *ret set
 * to an error).
 */
static inline AVBufferRef *vp_export_alloc_buffer(const VPExportFiles *files, int32_t id,
                                                  int qp_w, int qp_h,
                                                  int mv_w, int mv_h,
                                                  int bits_w, int bits_h,
                                                  int bits_block_size, int *ret)
{
    AVBufferRef *buf;
    VPExportHeader *h;
    size_t size = sizeof(*h), qp_offset, mv_offset, bits_offset;

    *ret = 0;
    if (!files->qp)
        qp_w = qp_h = 0;
    if (!files->mv)
        mv_w = mv_h = 0;
    if (!files->bits)
        bits_w = bits_h = 0;
    if (!qp_w && !mv_w && !bits_w)
        return NULL;

    qp_offset   = size;
    size       += ((size_t)qp_w * qp_h * sizeof(int16_t) + 3) & ~(size_t)3;
    mv_offset   = size;
    size       += (size_t)mv_w * mv_h * sizeof(VPExportMV);
    bits_offset = size;
    size       += (size_t)bits_w * bits_h * sizeof(VPExportBits);

    buf = av_buffer_allocz(size);
    if (!buf) {
        *ret = AVERROR(ENOMEM);
        return NULL;
    }
    h = (VPExportHeader *)buf->data;
    h->id = id;
    h->qp_w = qp_w;
    h->qp_h = qp_h;
    h->mv_w = mv_w;
    h->mv_h = mv_h;
    h->bits_w = bits_w;
    h->bits_h = bits_h;
    h->bits_block_size = bits_block_size;
    h->qp_offset = qp_offset;
    h->mv_offset = mv_offset;
    h->bits_offset = bits_offset;
    for (int i = 0; i < mv_w * mv_h; i++)
        vp_export_mv_none(&vp_export_mv(h)[i]);
    return buf;
}

/**
 * Attach a zeroed export buffer to the frame, see vp_export_alloc_buffer().
 */
static inline VPExportHeader *vp_export_alloc(AVFrame *f, const VPExportFiles *files,
                                              int32_t id,
                                              int qp_w, int qp_h,
                                              int mv_w, int mv_h,
                                              int bits_w, int bits_h,
                                              int bits_block_size, int *ret)
{
    AVBufferRef *buf = vp_export_alloc_buffer(files, id, qp_w, qp_h, mv_w, mv_h,
                                              bits_w, bits_h, bits_block_size, ret);
    AVFrameSideData *sd;

    if (!buf)
        return NULL;
    av_frame_remove_side_data(f, AV_FRAME_DATA_VIDEOPARSER_BLOCKS);
    sd = av_frame_new_side_data_from_buf(f, AV_FRAME_DATA_VIDEOPARSER_BLOCKS, buf);
    if (!sd) {
        av_buffer_unref(&buf);
        *ret = AVERROR(ENOMEM);
        return NULL;
    }
    return (VPExportHeader *)sd->data;
}

static inline VPExportHeader *vp_export_get(const AVFrame *f)
{
    AVFrameSideData *sd = av_frame_get_side_data(f, AV_FRAME_DATA_VIDEOPARSER_BLOCKS);
    return sd ? (VPExportHeader *)sd->data : NULL;
}

static inline int vp_export_write_record(FILE *file, const int32_t *header, int nb_header,
                                         const void *data, size_t size)
{
    if (fwrite(header, sizeof(*header), nb_header, file) != nb_header ||
        (size && fwrite(data, 1, size, file) != size))
        return AVERROR(EIO);
    return 0;
}

/**
 * Write the exports of an output frame. A frame without export data (h is
 * NULL, for example after a decoding error) gives an empty record, so that the
 * records stay aligned with the output frames.
 */
static inline int vp_export_write_header(const VPExportFiles *files, VPExportHeader *h)
{
    VPExportHeader empty = { 0 };
    int ret;

    if (!h)
        h = &empty;

    if (files->qp) {
        const int32_t hdr[3] = { h->id, h->qp_w, h->qp_h };
        ret = vp_export_write_record(files->qp, hdr, 3, vp_export_qp(h),
                                     (size_t)h->qp_w * h->qp_h * sizeof(int16_t));
        if (ret < 0)
            return ret;
    }
    if (files->mv) {
        const int32_t hdr[3] = { h->id, h->mv_w, h->mv_h };
        ret = vp_export_write_record(files->mv, hdr, 3, vp_export_mv(h),
                                     (size_t)h->mv_w * h->mv_h * sizeof(VPExportMV));
        if (ret < 0)
            return ret;
    }
    if (files->bits) {
        const int32_t hdr[4] = { h->id, h->bits_w, h->bits_h, h->bits_block_size };
        ret = vp_export_write_record(files->bits, hdr, 4, vp_export_bits(h),
                                     (size_t)h->bits_w * h->bits_h * sizeof(VPExportBits));
        if (ret < 0)
            return ret;
    }
    return 0;
}

/**
 * Write the exports attached to an output frame.
 */
static inline int vp_export_write(const VPExportFiles *files, const AVFrame *f)
{
    return vp_export_write_header(files, vp_export_get(f));
}

/**
 * Open the export files whose path is set.
 */
static inline int vp_export_open(VPExportFiles *files, const char *qp_path,
                                 const char *mv_path, const char *bits_path)
{
    if (qp_path && !(files->qp = fopen(qp_path, "wb")))
        return AVERROR(errno);
    if (mv_path && !(files->mv = fopen(mv_path, "wb")))
        return AVERROR(errno);
    if (bits_path && !(files->bits = fopen(bits_path, "wb")))
        return AVERROR(errno);
    return 0;
}

static inline void vp_export_close(VPExportFiles *files)
{
    if (files->qp)
        fclose(files->qp);
    if (files->mv)
        fclose(files->mv);
    if (files->bits)
        fclose(files->bits);
    files->qp = files->mv = files->bits = NULL;
}

#endif /* AVCODEC_VIDEOPARSER_EXPORT_H */
