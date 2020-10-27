/*****************************************************************************
 *
 * This file is part of Mapnik (c++ mapping toolkit)
 *
 * Copyright (C) 2017 Artem Pavlenko
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *****************************************************************************/

// mapnik
#include <mapnik/text/renderer.hpp>
#include <mapnik/grid/grid.hpp>
#include <mapnik/text/text_properties.hpp>
#include <mapnik/font_engine_freetype.hpp>
#include <mapnik/image_compositing.hpp>
#include <mapnik/image_scaling.hpp>
#include <mapnik/text/face.hpp>
#include <mapnik/image_util.hpp>
#include <mapnik/image_any.hpp>
#include <mapnik/agg_rasterizer.hpp>

#pragma GCC diagnostic push
#include <mapnik/warning_ignore_agg.hpp>
#include "agg_rendering_buffer.h"
#include "agg_pixfmt_rgba.h"
#include "agg_color_rgba.h"
#include "agg_scanline_u.h"
#include "agg_image_filters.h"
#include "agg_trans_bilinear.h"
#include "agg_span_allocator.h"
#include "agg_image_accessors.h"
#include "agg_span_image_filter_rgba.h"
#include "agg_renderer_base.h"
#include "agg_renderer_scanline.h"
#pragma GCC diagnostic pop

namespace mapnik
{

text_renderer::text_renderer (halo_rasterizer_e rasterizer, composite_mode_e comp_op,
                              composite_mode_e halo_comp_op, double scale_factor, stroker_ptr stroker)
    : rasterizer_(rasterizer),
      comp_op_(comp_op),
      halo_comp_op_(halo_comp_op),
      scale_factor_(scale_factor),
      glyphs_(),
      stroker_(stroker),
      transform_(),
      halo_transform_()
{}

void text_renderer::set_transform(agg::trans_affine const& transform)
{
    transform_ = transform;
}

void text_renderer::set_halo_transform(agg::trans_affine const& halo_transform)
{
    halo_transform_ = halo_transform;
}

void text_renderer::prepare_glyphs(glyph_positions const& positions)
{
    FT_Matrix matrix;
    FT_Vector pen;
    FT_Error  error;

    glyphs_.clear();
    glyphs_.reserve(positions.size());

    for (auto const& glyph_pos : positions)
    {
        glyph_info const& glyph = glyph_pos.glyph;
        FT_Int32 load_flags = FT_LOAD_DEFAULT | FT_LOAD_NO_HINTING;

        FT_Face face = glyph.face->get_face();
        if (glyph.face->is_color())
        {
            load_flags |= FT_LOAD_COLOR ;
            if (face->num_fixed_sizes > 0)
            {
                int scaled_size = static_cast<int>(glyph.format->text_size * scale_factor_);
                int best_match = 0;
                int diff = std::abs(scaled_size - face->available_sizes[0].width);
                for (int i = 1; i < face->num_fixed_sizes; ++i)
                {
                    int ndiff = std::abs(scaled_size - face->available_sizes[i].height);
                    if (ndiff < diff)
                    {
                        best_match = i;
                        diff = ndiff;
                    }
                }
                error = FT_Select_Size(face, best_match);
            }
        }
        else
        {
            glyph.face->set_character_sizes(glyph.format->text_size * scale_factor_);
        }

        double size = glyph.format->text_size * scale_factor_;
        matrix.xx = static_cast<FT_Fixed>( glyph_pos.rot.cos * 0x10000L);
        matrix.xy = static_cast<FT_Fixed>(-glyph_pos.rot.sin * 0x10000L);
        matrix.yx = static_cast<FT_Fixed>( glyph_pos.rot.sin * 0x10000L);
        matrix.yy = static_cast<FT_Fixed>( glyph_pos.rot.cos * 0x10000L);

        pixel_position pos = glyph_pos.pos + glyph.offset.rotate(glyph_pos.rot);
        pen.x = static_cast<FT_Pos>(pos.x * 64);
        pen.y = static_cast<FT_Pos>(pos.y * 64);

        FT_Set_Transform(face, &matrix, &pen);
        error = FT_Load_Glyph(face, glyph.glyph_index, load_flags);
        if (error) continue;
        FT_Glyph image;
        error = FT_Get_Glyph(face->glyph, &image);
        if (error) continue;
        box2d<double> bbox(0, glyph_pos.glyph.ymin(), glyph_pos.glyph.advance(), glyph_pos.glyph.ymax());
        glyphs_.emplace_back(image, *glyph.format, pos, glyph_pos.rot, size, bbox);
    }
}

template <typename T>
void composite_bitmap(T & pixmap, FT_Bitmap *bitmap, unsigned rgba, int x, int y, double opacity, composite_mode_e comp_op)
{
    int x_max = x + bitmap->width;
    int y_max = y + bitmap->rows;

    for (int i = x, p = 0; i < x_max; ++i, ++p)
    {
        for (int j = y, q = 0; j < y_max; ++j, ++q)
        {
            unsigned gray = bitmap->buffer[q * bitmap->width + p];
            if (gray)
            {
                mapnik::composite_pixel(pixmap, comp_op, i, j, rgba, gray, opacity);
            }
        }
    }
}

bool compare_luma(color c1, color c2)
{
    return c1.luma() < c2.luma();
}


void halo_bgsmooth_vacuum_candidates(
    std::list<color> & halo_color_candidates,
    size_t candidate_count_target,
    double halo_bgsmooth_outlier_lotrim,
    double halo_bgsmooth_outlier_hitrim)
{
    halo_color_candidates.sort(compare_luma);
    size_t trim_size = halo_color_candidates.size() - candidate_count_target;
    if (trim_size <= 0) return;
    size_t trim_begin_size = halo_bgsmooth_outlier_lotrim * trim_size;
    size_t trim_end_size = std::min(trim_size - trim_begin_size, (size_t)(halo_bgsmooth_outlier_hitrim * trim_size));
    for (size_t i = 0; i < trim_begin_size; ++i)
    {
        halo_color_candidates.pop_front();
    }
    for (size_t i = 0; i < trim_end_size; ++i)
    {
        halo_color_candidates.pop_back();
    }
}

template <typename T>
void halo_bgsmooth_acc_candidates_with_outlier_trimming(
    T & pixmap,
    FT_Bitmap *bitmap,
    int x,
    int y,
    std::list<color> & halo_color_candidates,
    std::uint8_t halo_bgsmooth_min_luma,
    std::uint8_t halo_bgsmooth_max_luma,
    double halo_bgsmooth_outlier_lotrim,
    double halo_bgsmooth_outlier_hitrim)
{
    size_t candidate_count_target = 512;
    size_t candidate_count_vacuum_threshold = 4096;

    int x_max = x + bitmap->width;
    int y_max = y + bitmap->rows;
    color src_color = color(0,0,0);
    std::uint8_t src_color_luma = 0;
    for (int i = x, p = 0; i < x_max; ++i, ++p)
    {
        for (int j = y, q = 0; j < y_max; ++j, ++q)
        {
            unsigned gray=bitmap->buffer[q*bitmap->width+p];
            if (gray && mapnik::check_bounds(pixmap, i, j))
            {
                src_color = mapnik::get_pixel<color>(pixmap, i, j);
                src_color_luma = src_color.luma();
                if (src_color.alpha() > 0
                  && src_color_luma >= halo_bgsmooth_min_luma
                  && src_color_luma <= halo_bgsmooth_max_luma)
                {
                    halo_color_candidates.push_back(src_color);

                    if (halo_color_candidates.size() > candidate_count_vacuum_threshold)
                    {
                        halo_bgsmooth_vacuum_candidates(
                            halo_color_candidates,
                            candidate_count_target,
                            halo_bgsmooth_outlier_lotrim,
                            halo_bgsmooth_outlier_hitrim);
                    }
                }
            }
        }
    }
}
template <typename T>
void halo_bgsmooth_acc_candidates(
    T & pixmap,
    FT_Bitmap *bitmap,
    int x,
    int y,
    std::list<color> & halo_color_candidates,
    std::uint8_t halo_bgsmooth_min_luma,
    std::uint8_t halo_bgsmooth_max_luma)
{
    int r_acc = 0;
    int g_acc = 0;
    int b_acc = 0;
    int acc_count = 0;

    int x_max = x + bitmap->width;
    int y_max = y + bitmap->rows;
    color src_color = color(0,0,0);
    std::uint8_t src_color_luma = 0;
    for (int i = x, p = 0; i < x_max; ++i, ++p)
    {
        for (int j = y, q = 0; j < y_max; ++j, ++q)
        {
            unsigned gray=bitmap->buffer[q*bitmap->width+p];
            if (gray && mapnik::check_bounds(pixmap, i, j))
            {
                src_color = mapnik::get_pixel<color>(pixmap, i, j);
                src_color_luma = src_color.luma();
                if (src_color.alpha() > 0
                  && src_color_luma >= halo_bgsmooth_min_luma
                  && src_color_luma <= halo_bgsmooth_max_luma)
                {
                    r_acc += src_color.red();
                    g_acc += src_color.green();
                    b_acc += src_color.blue();
                    ++acc_count;
                }
            }
        }
    }
    if (!acc_count) return;
    halo_color_candidates.push_back(color(
        (std::uint8_t)(r_acc / acc_count),
        (std::uint8_t)(g_acc / acc_count),
        (std::uint8_t)(b_acc / acc_count)
    ));
}

unsigned halo_bgsmooth_compute_color(
    unsigned default_rgba,
    std::list<color> & halo_color_candidates,
    double halo_bgsmooth_outlier_lotrim,
    double halo_bgsmooth_outlier_hitrim)
{
    int halo_color_candidates_count = halo_color_candidates.size();
    if (!halo_color_candidates_count) return default_rgba;

    int i_start = 0;
    int i_end = halo_color_candidates_count;
    // toss out x% on each end
    if (halo_bgsmooth_outlier_hitrim > 0 || halo_bgsmooth_outlier_lotrim > 0) {
        halo_color_candidates.sort(compare_luma);
        i_start = std::min(halo_color_candidates_count - 1, (int)(halo_color_candidates_count * halo_bgsmooth_outlier_lotrim));
        i_end = std::max(0, halo_color_candidates_count - (int)(halo_color_candidates_count * halo_bgsmooth_outlier_hitrim));
        if (i_start > i_end) {
            i_start = i_end = (i_start + i_end) / 2;
        }
        if (i_start == i_end) {
            i_start = std::max(0, i_start - 1);
            i_end = i_start + 1;
        }
    }

    // average colors
    int r_acc = 0;
    int g_acc = 0;
    int b_acc = 0;
    int acc_count = i_end - i_start;
    int i = 0;
    for (auto const& halo_color_candidate : halo_color_candidates) {
        if (i < i_start) { i++; continue; }
        if (i >= i_end) break;
        r_acc += halo_color_candidate.red();
        g_acc += halo_color_candidate.green();
        b_acc += halo_color_candidate.blue();
        i++;
    }

    if (!acc_count) return default_rgba;
    return color(
        (std::uint8_t)(r_acc / acc_count),
        (std::uint8_t)(g_acc / acc_count),
        (std::uint8_t)(b_acc / acc_count)
    ).rgba();
}

template <typename T>
agg_text_renderer<T>::agg_text_renderer (pixmap_type & pixmap,
                                         halo_rasterizer_e rasterizer,
                                         composite_mode_e comp_op,
                                         composite_mode_e halo_comp_op,
                                         double scale_factor,
                                         stroker_ptr stroker)
    : text_renderer(rasterizer, comp_op, halo_comp_op, scale_factor, stroker), pixmap_(pixmap)
{}

template <typename T>
void agg_text_renderer<T>::render(glyph_positions const& pos)
{
    prepare_glyphs(pos);

    FT_Error  error;
    FT_Vector start;
    FT_Vector start_halo;
    int height = pixmap_.height();
    pixel_position const& base_point = pos.get_base_point();

    start.x =  static_cast<FT_Pos>(base_point.x * (1 << 6));
    start.y =  static_cast<FT_Pos>((height - base_point.y) * (1 << 6));
    start_halo = start;
    start.x += transform_.tx * 64;
    start.y += transform_.ty * 64;
    start_halo.x += halo_transform_.tx * 64;
    start_halo.y += halo_transform_.ty * 64;

    FT_Matrix halo_matrix;
    halo_matrix.xx = halo_transform_.sx  * 0x10000L;
    halo_matrix.xy = halo_transform_.shx * 0x10000L;
    halo_matrix.yy = halo_transform_.sy  * 0x10000L;
    halo_matrix.yx = halo_transform_.shy * 0x10000L;

    FT_Matrix matrix;
    matrix.xx = transform_.sx  * 0x10000L;
    matrix.xy = transform_.shx * 0x10000L;
    matrix.yy = transform_.sy  * 0x10000L;
    matrix.yx = transform_.shy * 0x10000L;

    // default formatting
    double halo_radius = 0;
    color black(0,0,0);
    unsigned fill = black.rgba();
    unsigned halo_fill = black.rgba();
    double text_opacity = 1.0;
    double halo_opacity = 1.0;
    halo_bgsmooth_group_e halo_bgsmooth_group = HALO_BGSMOOTH_GROUP_CHARACTER;

    std::vector<FT_BitmapGlyph> pending_halo_glyph_bitmaps;
    std::vector<unsigned> pending_halo_glyph_rgbas;
    std::vector<glyph_t> pending_halo_glyphs;
    std::list<color> halo_color_candidates;
    std::vector<glyph_position>::const_iterator posptr = pos.begin();

    for (auto const& glyph : glyphs_)
    {
        glyph_info const& glyph_info_ = (*posptr++).glyph;

        bool did_buffer_glyph = false;
        halo_fill = glyph.properties.halo_fill.rgba();
        halo_opacity = glyph.properties.halo_opacity;
        halo_radius = glyph.properties.halo_radius * scale_factor_;
        halo_bgsmooth_group = glyph.properties.halo_bgsmooth_group;

        // make sure we've got reasonable values.
        if (halo_radius <= 0.0 || halo_radius > 1024.0) continue;

        FT_Glyph g;
        error = FT_Glyph_Copy(glyph.image, &g);
        if (!error)
        {
          FT_Glyph_Transform(g, &halo_matrix, &start_halo);

          if (rasterizer_ == HALO_RASTERIZER_FULL)
          {
              stroker_->init(halo_radius);
              FT_Glyph_Stroke(&g, stroker_->get(), 1);
              error = FT_Glyph_To_Bitmap(&g, FT_RENDER_MODE_NORMAL, 0, 1);
              if (!error)
              {
                  FT_BitmapGlyph bit = reinterpret_cast<FT_BitmapGlyph>(g);
                  if (bit->bitmap.pixel_mode != FT_PIXEL_MODE_BGRA)
                  {
                      if (glyph.properties.halo_bgsmooth) {
                          if (glyph.properties.halo_bgsmooth_outlier_lotrim > 0 || glyph.properties.halo_bgsmooth_outlier_hitrim > 0)
                          {
                              halo_bgsmooth_acc_candidates_with_outlier_trimming(
                                  pixmap_,
                                  &bit->bitmap,
                                  bit->left,
                                  height - bit->top,
                                  halo_color_candidates,
                                  glyph.properties.halo_bgsmooth_min.luma(),
                                  glyph.properties.halo_bgsmooth_max.luma(),
                                  glyph.properties.halo_bgsmooth_outlier_lotrim,
                                  glyph.properties.halo_bgsmooth_outlier_hitrim);
                          } else {
                              halo_bgsmooth_acc_candidates(
                                  pixmap_,
                                  &bit->bitmap,
                                  bit->left,
                                  height - bit->top,
                                  halo_color_candidates,
                                  glyph.properties.halo_bgsmooth_min.luma(),
                                  glyph.properties.halo_bgsmooth_max.luma());
                          }

                          if (halo_bgsmooth_group == HALO_BGSMOOTH_GROUP_CHARACTER || halo_bgsmooth_group == HALO_BGSMOOTH_GROUP_NONE) {
                              unsigned final_character_color = halo_bgsmooth_compute_color(
                                  halo_fill,
                                  halo_color_candidates,
                                  glyph.properties.halo_bgsmooth_outlier_lotrim,
                                  glyph.properties.halo_bgsmooth_outlier_hitrim);
                              pending_halo_glyph_rgbas.push_back(final_character_color);
                              halo_color_candidates.clear();
                          }

                          pending_halo_glyph_bitmaps.push_back(bit);
                          pending_halo_glyphs.push_back(glyph);
                          did_buffer_glyph = true;
                      } else {
                          composite_bitmap(pixmap_,
                                           &bit->bitmap,
                                           halo_fill,
                                           bit->left,
                                           height - bit->top,
                                           halo_opacity,
                                           halo_comp_op_);
                      }
                  }
              }
          }
        }
        else
        {
            error = FT_Glyph_To_Bitmap(&g, FT_RENDER_MODE_NORMAL, 0, 1);
            if (!error)
            {
                FT_BitmapGlyph bit = reinterpret_cast<FT_BitmapGlyph>(g);
                if (bit->bitmap.pixel_mode == FT_PIXEL_MODE_BGRA)
                {
                    pixel_position render_pos(base_point);
                    image_rgba8 glyph_image(render_glyph_image(glyph,
                                                               bit->bitmap,
                                                               transform_,
                                                               render_pos));
                    const constexpr std::size_t pixel_size = sizeof(image_rgba8::pixel_type);
                    render_halo<pixel_size>(glyph_image.bytes(),
                                            glyph_image.width(),
                                            glyph_image.height(),
                                            halo_fill,
                                            render_pos.x, render_pos.y,
                                            halo_radius,
                                            halo_opacity,
                                            halo_comp_op_);
                }
                else
                {
                    render_halo<1>(bit->bitmap.buffer,
                                   bit->bitmap.width,
                                   bit->bitmap.rows,
                                   halo_fill,
                                   bit->left,
                                   height - bit->top,
                                   halo_radius,
                                   halo_opacity,
                                   halo_comp_op_);
                }
            }
        }

        if (did_buffer_glyph) {
            if (halo_bgsmooth_group == HALO_BGSMOOTH_GROUP_WORD) {
              unsigned glyph_index__space = FT_Get_Char_Index(glyph_info_.face->get_face(), 32);
              unsigned glyph_index__newline = FT_Get_Char_Index(glyph_info_.face->get_face(), 10);
              unsigned glyph_index__nbsp = FT_Get_Char_Index(glyph_info_.face->get_face(), 0x00A0);
              if (glyph_info_.glyph_index == glyph_index__space /* SPACE */ || glyph_info_.glyph_index == glyph_index__nbsp /* NBSP */ || glyph_info_.glyph_index == glyph_index__newline /* NEW_LINE */) {
                    unsigned word_color = halo_bgsmooth_compute_color(
                        halo_fill,
                        halo_color_candidates,
                        glyph.properties.halo_bgsmooth_outlier_lotrim,
                        glyph.properties.halo_bgsmooth_outlier_hitrim);
                    while (pending_halo_glyph_rgbas.size() < pending_halo_glyph_bitmaps.size()) {
                        pending_halo_glyph_rgbas.push_back(word_color);
                    }
                    halo_color_candidates.clear();
                }
            } else if (halo_bgsmooth_group == HALO_BGSMOOTH_GROUP_LINE) {
                unsigned glyph_index__newline = FT_Get_Char_Index(glyph_info_.face->get_face(), 10);
                if (glyph_info_.glyph_index == glyph_index__newline /* NEW_LINE */) {
                    unsigned line_color = halo_bgsmooth_compute_color(
                        halo_fill,
                        halo_color_candidates,
                        glyph.properties.halo_bgsmooth_outlier_lotrim,
                        glyph.properties.halo_bgsmooth_outlier_hitrim);
                    while (pending_halo_glyph_rgbas.size() < pending_halo_glyph_bitmaps.size()) {
                        pending_halo_glyph_rgbas.push_back(line_color);
                    }
                    halo_color_candidates.clear();
                }
            }
        }

        if (!did_buffer_glyph) FT_Done_Glyph(g);
    }

    // fill pending halo colors
    if (pending_halo_glyph_rgbas.size() < pending_halo_glyph_bitmaps.size()) {
      glyph_t last_glyph = glyphs_[glyphs_.size() - 1];
      unsigned final_color = halo_bgsmooth_compute_color(
          halo_fill,
          halo_color_candidates,
          last_glyph.properties.halo_bgsmooth_outlier_lotrim,
          last_glyph.properties.halo_bgsmooth_outlier_hitrim);
      while (pending_halo_glyph_rgbas.size() < pending_halo_glyph_bitmaps.size()) {
          pending_halo_glyph_rgbas.push_back(final_color);
      }
    }

    // render buffered halos
    for (int i = 0, n = pending_halo_glyph_bitmaps.size(); i < n; ++i) {
        FT_BitmapGlyph bit = pending_halo_glyph_bitmaps[i];
        glyph_t glyph = pending_halo_glyphs[i];
        unsigned halo_fill = pending_halo_glyph_rgbas[i];
        composite_bitmap(pixmap_,
                         &bit->bitmap,
                         halo_fill,
                         bit->left,
                         height - bit->top,
                         glyph.properties.halo_opacity,
                         halo_comp_op_);

        FT_Done_Glyph(reinterpret_cast<FT_Glyph>(bit));
    }

    // render actual text
    for (auto & glyph : glyphs_)
    {
        fill = glyph.properties.fill.rgba();
        text_opacity = glyph.properties.text_opacity;

        FT_Glyph_Transform(glyph.image, &matrix, &start);
        error = 0;
        if ( glyph.image->format != FT_GLYPH_FORMAT_BITMAP )
        {
            error = FT_Glyph_To_Bitmap(&glyph.image ,FT_RENDER_MODE_NORMAL, 0, 1);
        }
        if (error == 0)
        {
            FT_BitmapGlyph bit = reinterpret_cast<FT_BitmapGlyph>(glyph.image);
            int pixel_mode = bit->bitmap.pixel_mode;
            if (pixel_mode == FT_PIXEL_MODE_BGRA)
            {
                int x = base_point.x + glyph.pos.x;
                int y = base_point.y - glyph.pos.y;
                agg::trans_affine transform(
                    glyph_transform(transform_,
                                    bit->bitmap.rows,
                                    x, y,
                                    -glyph.rot.angle(),
                                    glyph.bbox));
                composite_color_glyph(pixmap_,
                                      bit->bitmap,
                                      transform,
                                      text_opacity,
                                      comp_op_);
            }
            else
            {
                composite_bitmap(pixmap_,
                                 &bit->bitmap,
                                 fill,
                                 bit->left,
                                 height - bit->top,
                                 text_opacity,
                                 comp_op_);
            }
        }
        FT_Done_Glyph(glyph.image);
    }

}

template <typename T>
void grid_text_renderer<T>::render(glyph_positions const& pos, value_integer feature_id)
{
    prepare_glyphs(pos);
    FT_Error  error;
    FT_Vector start;
    unsigned height = pixmap_.height();
    pixel_position const& base_point = pos.get_base_point();
    start.x =  static_cast<FT_Pos>(base_point.x * (1 << 6));
    start.y =  static_cast<FT_Pos>((height - base_point.y) * (1 << 6));
    start.x += transform_.tx * 64;
    start.y += transform_.ty * 64;

    // now render transformed glyphs
    double halo_radius = 0.0;
    FT_Matrix halo_matrix;
    halo_matrix.xx = halo_transform_.sx  * 0x10000L;
    halo_matrix.xy = halo_transform_.shx * 0x10000L;
    halo_matrix.yy = halo_transform_.sy  * 0x10000L;
    halo_matrix.yx = halo_transform_.shy * 0x10000L;
    for (auto & glyph : glyphs_)
    {
        halo_radius = glyph.properties.halo_radius * scale_factor_;
        FT_Glyph_Transform(glyph.image, &halo_matrix, &start);
        error = FT_Glyph_To_Bitmap(&glyph.image, FT_RENDER_MODE_NORMAL, 0, 1);
        if (!error)
        {
            FT_BitmapGlyph bit = reinterpret_cast<FT_BitmapGlyph>(glyph.image);
            if (bit->bitmap.pixel_mode == FT_PIXEL_MODE_BGRA)
            {
                pixel_position render_pos(base_point);
                image_rgba8 glyph_image(render_glyph_image(glyph,
                                                           bit->bitmap,
                                                           transform_,
                                                           render_pos));
                const constexpr std::size_t pixel_size = sizeof(image_rgba8::pixel_type);
                render_halo_id<pixel_size>(glyph_image.bytes(),
                                           glyph_image.width(),
                                           glyph_image.height(),
                                           feature_id,
                                           render_pos.x, render_pos.y,
                                           static_cast<int>(halo_radius));
            }
            else
            {
                render_halo_id<1>(bit->bitmap.buffer,
                                  bit->bitmap.width,
                                  bit->bitmap.rows,
                                  feature_id,
                                  bit->left,
                                  height - bit->top,
                                  static_cast<int>(halo_radius));
            }
        }
        FT_Done_Glyph(glyph.image);
    }
}


template <typename T>
template <std::size_t PixelWidth>
void agg_text_renderer<T>::render_halo(unsigned char *buffer,
                                       unsigned width,
                                       unsigned height,
                                       unsigned rgba,
                                       int x1,
                                       int y1,
                                       double halo_radius,
                                       double opacity,
                                       composite_mode_e comp_op)
{
    if (halo_radius < 1.0)
    {
        for (unsigned x = 0; x < width; ++x)
        {
            for (unsigned y = 0; y < height; ++y)
            {
                int gray = buffer[(y * width + x) * PixelWidth + PixelWidth - 1];
                if (gray)
                {
                    mapnik::composite_pixel(pixmap_, comp_op, x+x1-1, y+y1-1, rgba, gray*halo_radius*halo_radius, opacity);
                    mapnik::composite_pixel(pixmap_, comp_op, x+x1,   y+y1-1, rgba, gray*halo_radius, opacity);
                    mapnik::composite_pixel(pixmap_, comp_op, x+x1+1, y+y1-1, rgba, gray*halo_radius*halo_radius, opacity);

                    mapnik::composite_pixel(pixmap_, comp_op, x+x1-1, y+y1,   rgba, gray*halo_radius, opacity);
                    mapnik::composite_pixel(pixmap_, comp_op, x+x1,   y+y1,   rgba, gray, opacity);
                    mapnik::composite_pixel(pixmap_, comp_op, x+x1+1, y+y1,   rgba, gray*halo_radius, opacity);

                    mapnik::composite_pixel(pixmap_, comp_op, x+x1-1, y+y1+1, rgba, gray*halo_radius*halo_radius, opacity);
                    mapnik::composite_pixel(pixmap_, comp_op, x+x1,   y+y1+1, rgba, gray*halo_radius, opacity);
                    mapnik::composite_pixel(pixmap_, comp_op, x+x1+1, y+y1+1, rgba, gray*halo_radius*halo_radius, opacity);
                }
            }
        }
    }
    else
    {
        for (unsigned x = 0; x < width; ++x)
        {
            for (unsigned y = 0; y < height; ++y)
            {
                int gray = buffer[(y * width + x) * PixelWidth + PixelWidth - 1];
                if (gray)
                {
                    for (int n=-halo_radius; n <=halo_radius; ++n)
                        for (int m=-halo_radius; m <= halo_radius; ++m)
                            mapnik::composite_pixel(pixmap_, comp_op, x+x1+m, y+y1+n, rgba, gray, opacity);
                }
            }
        }
    }
}

template <typename T>
template <std::size_t PixelWidth>
void grid_text_renderer<T>::render_halo_id(unsigned char *buffer,
                                           unsigned width,
                                           unsigned height,
                                           mapnik::value_integer feature_id,
                                           int x1,
                                           int y1,
                                           int halo_radius)
{
    for (unsigned x = 0; x < width; ++x)
    {
        for (unsigned y = 0; y < height; ++y)
        {
            int gray = buffer[(y * width + x) * PixelWidth + PixelWidth - 1];
            if (gray)
            {
                for (int n = -halo_radius; n <=halo_radius; ++n)
                    for (int m = -halo_radius; m <= halo_radius; ++m)
                        pixmap_.setPixel(x+x1+m,y+y1+n,feature_id);
            }
        }
    }
}

template <typename T>
grid_text_renderer<T>::grid_text_renderer(pixmap_type &pixmap,
                                          composite_mode_e comp_op,
                                          double scale_factor)
    : text_renderer(HALO_RASTERIZER_FAST, comp_op, src_over, scale_factor),
      pixmap_(pixmap) {}

template class agg_text_renderer<image_rgba8>;
template class grid_text_renderer<grid>;

} // namespace mapnik
