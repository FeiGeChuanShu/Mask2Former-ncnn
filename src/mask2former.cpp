#include "cpu.h"
#include "net.h"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
#define MAX_STRIDE 32
struct Object {
    int label;
    float prob;
    cv::Mat cv_mask;
};

class ms_attn : public ncnn::Layer {
  public:
    ms_attn() {
        one_blob_only = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const {
        const ncnn::Mat& value_spatial_shapes = bottom_blobs[0];
        const ncnn::Mat& value = bottom_blobs[1];
        const ncnn::Mat& sampling_locations1 = bottom_blobs[3];
        const ncnn::Mat& sampling_locations2 = bottom_blobs[4];
        const ncnn::Mat& attention_weights = bottom_blobs[2];

        int M  = value.h;
        int D  = value.w;
        int Lq = sampling_locations1.c;
        int L  = sampling_locations1.h;
        int P  = sampling_locations1.w;

        int outw = M * D;
        int outh = Lq;

        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(outw, outh, 4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* value_spatial_shapes_ptr = value_spatial_shapes.channel(0);
        const float* sampling_locations1_ptr = (float*)sampling_locations1.data;
        const float* sampling_locations2_ptr = (float*)sampling_locations2.data;
        const float* value_ptr = (float*)value.data;
        const float* attention_weights_ptr = (float*)attention_weights.data;
        float* top_blob_ptr = (float*)top_blob.data;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < Lq; j++) {
            int value_offset = 0;
            int w = 0;
            int h = 0;
            float x = 0.f;
            float y = 0.f;
            float value_attention[8][32];
            float xy_grid[2][8];
            int x0[8];
            int y0[8];
            int x1[8];
            int y1[8];
            float Ia[8][32];
            float Ib[8][32];
            float Ic[8][32];
            float Id[8][32];
            memset(value_attention, 0.f, sizeof(float) * 8 * 32);
            memset(xy_grid, 0.f, sizeof(float) * 8 * 2);
            memset(Ia, 0.f, sizeof(float) * 8 * 32);
            memset(Ib, 0.f, sizeof(float) * 8 * 32);
            memset(Ic, 0.f, sizeof(float) * 8 * 32);
            memset(Id, 0.f, sizeof(float) * 8 * 32);
            
            for (int i_d = 0; i_d < D; i_d++) {
                for (int i_m = 0; i_m < M; i_m++) {
                    value_attention[i_m][i_d] = 0.f;
                }
            }
            for (int i = 0; i < L; i++) {
                value_offset = 0;
                for (int ii = 0; ii < i; ii++) {
                    value_offset += static_cast<int>(value_spatial_shapes_ptr[ii * 2 + 0]) *
                                    static_cast<int>(value_spatial_shapes_ptr[ii * 2 + 1]);
                }

                for (int k = 0; k < P; k++) {
                    for (int i_m = 0; i_m < M; i_m++) {
                        w = value_spatial_shapes_ptr[i * 2 + 0];
                        h = value_spatial_shapes_ptr[i * 2 + 1];
                        x = sampling_locations1_ptr[j * M * L * P + i_m * L * P + i * P + k];
                        y = sampling_locations2_ptr[j * M * L * P + i_m * L * P + i * P + k];

                        xy_grid[0][i_m] = x * static_cast<float>(h) - 0.5; //x
                        xy_grid[1][i_m] = y * static_cast<float>(w) - 0.5; //y

                        x0[i_m] = static_cast<int>(std::floor(xy_grid[0][i_m])); //x0
                        x1[i_m] = x0[i_m] + 1;                     //x1
                        y0[i_m] = static_cast<int>(std::floor(xy_grid[1][i_m])); //y0
                        y1[i_m] = y0[i_m] + 1;                     //y1
                    }

                    for (int i_m = 0; i_m < M; i_m++) {
                        if (x0[i_m] < 0 || x0[i_m] >= h || y0[i_m] < 0 || y0[i_m] >= w) {
                            for (int i_d = 0; i_d < D; i_d++)
                                Ia[i_m][i_d] = 0.f;
                        } else {
                            for (int i_d = 0; i_d < D; i_d++) {
                                int idx = value_offset + y0[i_m] * h + x0[i_m];
                                Ia[i_m][i_d] = value_ptr[idx * M * D + i_m * D + i_d]; //x0_y0
                            }
                        }
                    }

                    for (int i_m = 0; i_m < M; i_m++) {
                        if (x0[i_m] < 0 || x0[i_m] >= h || y1[i_m] < 0 || y1[i_m] >= w) {
                            for (int i_d = 0; i_d < D; i_d++)
                                Ib[i_m][i_d] = 0.f;
                        } else {
                            for (int i_d = 0; i_d < D; i_d++) {
                                int idx = value_offset + y1[i_m] * h + x0[i_m];
                                Ib[i_m][i_d] = value_ptr[idx * M * D + i_m * D + i_d]; //x0_y1
                            }
                        }
                    }

                    for (int i_m = 0; i_m < M; i_m++) {
                        if (x1[i_m] < 0 || x1[i_m] >= h || y0[i_m] < 0 || y0[i_m] >= w) {
                            for (int i_d = 0; i_d < D; i_d++)
                                Ic[i_m][i_d] = 0.f;
                        } else {
                            for (int i_d = 0; i_d < D; i_d++) {
                                int idx = value_offset + y0[i_m] * h + x1[i_m];
                                Ic[i_m][i_d] = value_ptr[idx * M * D + i_m * D + i_d]; //x1_y0
                            }
                        }
                    }

                    for (int i_m = 0; i_m < M; i_m++) {
                        if (x1[i_m] < 0 || x1[i_m] >= h || y1[i_m] < 0 || y1[i_m] >= w) {
                            for (int i_d = 0; i_d < D; i_d++)
                                Id[i_m][i_d] = 0.f;
                        } else {
                            for (int i_d = 0; i_d < D; i_d++) {
                                int idx = value_offset + y1[i_m] * h + x1[i_m];
                                Id[i_m][i_d] = value_ptr[idx * M * D + i_m * D + i_d]; //x1_y1
                            }
                        }
                    }

                    for (int i_m = 0; i_m < M; i_m++) {
                        for (int i_d = 0; i_d < D; i_d++) {

                            float wa = (x1[i_m] - xy_grid[0][i_m]) * (y1[i_m] - xy_grid[1][i_m]);
                            float wb = (x1[i_m] - xy_grid[0][i_m]) * (xy_grid[1][i_m] - y0[i_m]);
                            float wc = (xy_grid[0][i_m] - x0[i_m]) * (y1[i_m] - xy_grid[1][i_m]);
                            float wd = (xy_grid[0][i_m] - x0[i_m]) * (xy_grid[1][i_m] - y0[i_m]);

                            value_attention[i_m][i_d] += (Ia[i_m][i_d] * wa    //x0_y0 * wa
                                                        + Ib[i_m][i_d] * wb  //x0_y1 * wb
                                                        + Ic[i_m][i_d] * wc  //x1_y0 * wc
                                                        + Id[i_m][i_d] * wd) //x1_y1 * wd
                                                       * attention_weights_ptr[j * M * L * P + i_m * L * P + i * P + k];
                        }
                    }

                    for (int i_m = 0; i_m < M; i_m++) {
                        for (int i_d = 0; i_d < D; i_d++) {
                            top_blob_ptr[j * M * D + i_m * D + i_d] = value_attention[i_m][i_d];
                        }
                    }
                }
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(ms_attn)

class gen_attn : public ncnn::Layer {
  public:
    gen_attn() {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const {
        int channel = bottom_blob.c;
        int size = bottom_blob.h * bottom_blob.w;

        top_blob.create(size, 100, 8, 4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        ncnn::Mat top_blob_ = ncnn::Mat(1, 100);
        ncnn::Mat tmp_blob = ncnn::Mat(size, 100);
        top_blob_.fill(1.0f * size);
        tmp_blob.fill(1.0f);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < channel; i++) {
            const float* ptr = bottom_blob.channel(i);
            float* ptr1 = tmp_blob.row(i);
            float* ptr0 = top_blob_.row(i);
            for (int j = 0; j < size; j++) {
                if (ptr[j] >= 0.5) {
                    ptr0[0] -= 1;
                    ptr1[j] = 0;
                }
            }
        }

        top_blob.fill(-std::numeric_limits<float>::infinity());
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < 8; i++) {
            float* attn = top_blob.channel(i);
            for (int j = 0; j < channel; j++) {
                float* ptr0 = top_blob_.row(j);
                float* ptr2 = tmp_blob.row(j);
                float* ptr3 = attn + j * size;

                if ((int)ptr0[0] == size) {
                    memset(ptr3, 0.f, sizeof(float) * size);
                } else {
                    for (int k = 0; k < size; k++) {
                        if (ptr2[k] == 0)
                            ptr3[k] = ptr2[k];
                    }
                }
            }
        }

        return 0;
    }
};
DEFINE_LAYER_CREATOR(gen_attn)

static void concat(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, int axis) {
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Concat");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, axis); // axis

    op->load_param(pd);

    op->create_pipeline(opt);

    op->forward(bottom_blobs, top_blobs, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void transpose(const ncnn::Mat& in, ncnn::Mat& out, const int& order_type) {
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = true;

    ncnn::Layer* op = ncnn::create_layer("Permute");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, order_type); // order_type

    op->load_param(pd);

    op->create_pipeline(opt);

    ncnn::Mat in_packed = in;
    {
        // resolve dst_elempack
        int dims = in.dims;
        int elemcount = 0;
        if (dims == 1)
            elemcount = in.elempack * in.w;
        if (dims == 2)
            elemcount = in.elempack * in.h;
        if (dims == 3)
            elemcount = in.elempack * in.c;

        int dst_elempack = 1;
        if (op->support_packing) {
#if (NCNN_AVX2 || NCNN_AVX)
            if (elemcount % 8 == 0 && (ncnn::cpu_support_x86_avx2() || ncnn::cpu_support_x86_avx()))
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
#endif
        }

        if (in.elempack != dst_elempack) {
            convert_packing(in, in_packed, dst_elempack, opt);
        }
    }

    // forward
    op->forward(in_packed, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Crop");

    // set param
    ncnn::ParamDict pd;

    ncnn::Mat axes = ncnn::Mat(1);
    axes.fill(axis);
    ncnn::Mat ends = ncnn::Mat(1);
    ends.fill(end);
    ncnn::Mat starts = ncnn::Mat(1);
    starts.fill(start);
    pd.set(9, starts); // start
    pd.set(10, ends);  // end
    pd.set(11, axes);  //axes

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

static void reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Reshape");

    // set param
    ncnn::ParamDict pd;

    pd.set(0, w); // start
    pd.set(1, h); // end
    if (d > 0)
        pd.set(11, d); //axes
    pd.set(2, c);      //axes
    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

static void interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Interp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2);     // resize_type
    pd.set(1, scale); // height_scale
    pd.set(2, scale); // width_scale
    pd.set(3, out_h); // height
    pd.set(4, out_w); // width

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void getTopk(const ncnn::Mat& cls_scores, int topk, std::vector<std::pair<float, int>>& scores,
                    std::vector<int>& labels_per_image, float mask_threshold) {
    int idx = 0;
    std::vector<std::pair<float, int>> vec;
    vec.resize(cls_scores.h * (cls_scores.w - 1));
    std::vector<int> labels;
    labels.resize(cls_scores.h * (cls_scores.w - 1));

    for (int i = 0; i < cls_scores.h; i++) {
        const float* ptr = cls_scores.row(i);
        for (int j = 0; j < cls_scores.w - 1; j++) {
            vec[idx] = std::make_pair(ptr[j], idx);
            idx++;
            labels[i * (cls_scores.w - 1) + j] = j;
        }
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int>>());
    for (int i = 0; i < topk; i++) {
        float score = vec[i].first;
        if (score < mask_threshold)
            continue;

        int index = vec[i].second;
        scores.push_back(std::pair<float, int>(score, std::floor(index / 80)));
        labels_per_image.push_back(labels[index]);
    }
}
static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}
static int position_embedding(ncnn::Mat& mask, int num_pos_feats, ncnn::Mat& pos) {
    ncnn::Mat y_embed = ncnn::Mat(mask.w, mask.h, mask.c);
    ncnn::Mat x_embed = ncnn::Mat(mask.w, mask.h, mask.c);

    for (int i = 0; i < mask.c; i++) {
        for (int j = 0; j < mask.h; j++) {
            float* mask_data = mask.channel(i).row(j);
            float* x_embed_data = x_embed.channel(i).row(j);
            for (int k = 0; k < mask.w; k++) {
                for (int l = k; l >= 0; l--)
                    x_embed_data[k] += mask_data[l];
            }
        }
        float* mask_data = mask.channel(i);
        for (int j = 0; j < mask.w; j++) {
            for (int k = 0; k < mask.h; k++) {
                float* y_embed_data = y_embed.channel(i).row(k);
                for (int l = k; l >= 0; l--)
                    y_embed_data[j] += mask_data[l * mask.w];
            }
        }
    }
    for (int i = 0; i < y_embed.c; i++) {
        for (int j = 0; j < y_embed.h; j++) {
            for (int k = 0; k < y_embed.w; k++) {
                y_embed[j * y_embed.w + k] = y_embed[j * y_embed.w + k] * 6.283185307179586 / (y_embed.row(y_embed.h - 1)[k] + 0.000001);
            }
        }
    }
    for (int i = 0; i < x_embed.c; i++) {
        for (int j = 0; j < x_embed.h; j++) {
            for (int k = 0; k < x_embed.w; k++) {
                x_embed[j * x_embed.w + k] = x_embed[j * x_embed.w + k] * 6.283185307179586 / (x_embed[j * x_embed.w + x_embed.w - 1] + 0.000001);
            }
        }
    }

    std::vector<float> dim_t;
    for (int i = 0; i < num_pos_feats; i++)
        dim_t.push_back(i);
    for (int i = 0; i < num_pos_feats; i++) {
        dim_t[i] = std::pow(10000.0, 2 * std::floor(dim_t[i] / 2) / num_pos_feats);
    }

    ncnn::Mat pos_x = ncnn::Mat(num_pos_feats, mask.w, mask.h);
    ncnn::Mat pos_y = ncnn::Mat(num_pos_feats, mask.w, mask.h);

    for (int i = 0; i < pos_x.c; i++) {
        float* pos_x_data = pos_x.channel(i);
        for (int j = 0; j < pos_x.h; j++) {
            for (int k = 0; k < pos_x.w; k++) {
                pos_x_data[j * pos_x.w + k] = x_embed[i * pos_x.h + j] / dim_t[k];
            }
        }
    }
    for (int i = 0; i < pos_y.c; i++) {
        float* pos_y_data = pos_y.channel(i);
        for (int j = 0; j < pos_y.h; j++) {
            for (int k = 0; k < pos_y.w; k++) {
                pos_y_data[j * pos_y.w + k] = y_embed[i * pos_y.h + j] / dim_t[k];
            }
        }
    }

    for (int i = 0; i < pos_x.c; i++) {
        float* data = pos_x.channel(i);
        for (int j = 0; j < pos_x.h; j++) {
            for (int k = 0; k < pos_x.w;) {
                data[j * pos_x.w + k] = std::sin(data[j * pos_x.w + k]);
                k += 2;
            }

            for (int k = 1; k < pos_x.w;) {
                data[j * pos_x.w + k] = std::cos(data[j * pos_x.w + k]);
                k += 2;
            }
        }
    }

    for (int i = 0; i < pos_y.c; i++) {
        float* data = pos_y.channel(i);
        for (int j = 0; j < pos_y.h; j++) {
            for (int k = 0; k < pos_y.w;) {
                data[j * pos_y.w + k] = std::sin(data[j * pos_y.w + k]);
                k += 2;
            }

            for (int k = 1; k < pos_y.w;) {
                data[j * pos_y.w + k] = std::cos(data[j * pos_y.w + k]);
                k += 2;
            }
        }
    }
    std::vector<ncnn::Mat> tops(1);
    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = pos_y;
    bottoms[1] = pos_x;
    concat(bottoms, tops, 2);
    pos = tops[0];
    transpose(pos, pos, 4);

    return 0;
}
static void get_reference_points(const float* spatial_shape, std::vector<ncnn::Mat>& reference_points) {
    size_t elemsize = sizeof(float);
    ncnn::Mat ref;
    std::vector<ncnn::Mat> reference_points_list;
    for (int i = 0; i < 3; i++) {
        // coord conv
        int pw = static_cast<int>(spatial_shape[i * 2 + 1]);
        int ph = static_cast<int>(spatial_shape[i * 2]);
        ref.create(pw, ph, 2, elemsize);
        float step_h = static_cast<float>(ph - 1) / (ph - 1);
        float step_w = static_cast<float>(pw - 1) / (pw - 1);
        for (int h = 0; h < ph; h++) {
            for (int w = 0; w < pw; w++) {

                ref.channel(0)[h * pw + w] = (0.5f + step_w * static_cast<float>(w)) / static_cast<float>(pw);
                ref.channel(1)[h * pw + w] = (0.5f + step_h * static_cast<float>(h)) / static_cast<float>(ph);
            }
        }

        ref = ref.reshape(pw * ph, 1, 2);
        transpose(ref, ref, 5);

        reference_points_list.push_back(ref);
    }

    std::vector<ncnn::Mat> reference_points_(3);
    concat(reference_points_list, reference_points_, 0);
    reference_points_[2] = reference_points_[0];
    reference_points_[1] = reference_points_[0];
    reference_points_[0] = reference_points_[0];

    concat(reference_points_, reference_points, 1);
    reference_points[0] = reference_points[0].reshape(reference_points[0].w, reference_points[0].h, reference_points[0].c, 1);
}
static void position_embedding_layer(const int& pos_w, const int& pos_h, ncnn::Mat& pos_em1,
                                     ncnn::Mat& pos_em2, ncnn::Mat& pos_em3, ncnn::Mat& pos1, ncnn::Mat& pos2, ncnn::Mat& pos3) {
    ncnn::Mat mask1 = ncnn::Mat(pos_w / 32, pos_h / 32, 2048);
    mask1.fill(1.0f);
    ncnn::Mat mask2 = ncnn::Mat(pos_w / 16, pos_h / 16, 1024);
    mask2.fill(1.0f);
    ncnn::Mat mask3 = ncnn::Mat(pos_w / 8, pos_h / 8, 512);
    mask3.fill(1.0f);
    position_embedding(mask1, 128, pos_em1);
    position_embedding(mask2, 128, pos_em2);
    position_embedding(mask3, 128, pos_em3);

    reshape(pos_em1, pos1, 1, 256, -1, 0);
    transpose(pos1, pos1, 4);
    reshape(pos_em2, pos2, 1, 256, -1, 0);
    transpose(pos2, pos2, 4);
    reshape(pos_em3, pos3, 1, 256, -1, 0);
    transpose(pos3, pos3, 4);
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects) {
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"};

    static const unsigned char colors[81][3] = {
        {56, 0, 255},
        {226, 255, 0},
        {0, 94, 255},
        {0, 37, 255},
        {0, 255, 94},
        {255, 226, 0},
        {0, 18, 255},
        {255, 151, 0},
        {170, 0, 255},
        {0, 255, 56},
        {255, 0, 75},
        {0, 75, 255},
        {0, 255, 169},
        {255, 0, 207},
        {75, 255, 0},
        {207, 0, 255},
        {37, 0, 255},
        {0, 207, 255},
        {94, 0, 255},
        {0, 255, 113},
        {255, 18, 0},
        {255, 0, 56},
        {18, 0, 255},
        {0, 255, 226},
        {170, 255, 0},
        {255, 0, 245},
        {151, 255, 0},
        {132, 255, 0},
        {75, 0, 255},
        {151, 0, 255},
        {0, 151, 255},
        {132, 0, 255},
        {0, 255, 245},
        {255, 132, 0},
        {226, 0, 255},
        {255, 37, 0},
        {207, 255, 0},
        {0, 255, 207},
        {94, 255, 0},
        {0, 226, 255},
        {56, 255, 0},
        {255, 94, 0},
        {255, 113, 0},
        {0, 132, 255},
        {255, 0, 132},
        {255, 170, 0},
        {255, 0, 188},
        {113, 255, 0},
        {245, 0, 255},
        {113, 0, 255},
        {255, 188, 0},
        {0, 113, 255},
        {255, 0, 0},
        {0, 56, 255},
        {255, 0, 113},
        {0, 255, 188},
        {255, 0, 94},
        {255, 0, 18},
        {18, 255, 0},
        {0, 255, 132},
        {0, 188, 255},
        {0, 245, 255},
        {0, 169, 255},
        {37, 255, 0},
        {255, 0, 151},
        {188, 0, 255},
        {0, 255, 37},
        {0, 255, 0},
        {255, 0, 170},
        {255, 0, 37},
        {255, 75, 0},
        {0, 0, 255},
        {255, 207, 0},
        {255, 0, 226},
        {255, 245, 0},
        {188, 255, 0},
        {0, 255, 18},
        {0, 255, 75},
        {0, 255, 151},
        {255, 56, 0},
        {245, 255, 0}};

    int color_index = 0;

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f\n", obj.label, obj.prob);

        const unsigned char* color = colors[color_index % 80];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        for (int y = 0; y < image.rows; y++) {
            uchar* image_ptr = image.ptr(y);
            const uchar* mask_ptr = obj.cv_mask.ptr<uchar>(y);
            for (int x = 0; x < image.cols; x++) {
                if (mask_ptr[x] > 0) {
                    image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + color[2] * 0.5);
                    image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + color[1] * 0.5);
                    image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + color[0] * 0.5);
                }
                image_ptr += 3;
            }
        }
    }
    cv::imwrite("result.jpg", image);
    cv::imshow("result", image);
    cv::waitKey(0);
}

static void decode_ins(const ncnn::Mat& output_mask, const ncnn::Mat& output_class,
                       const float& mask_threshold, const int& img_w, const int& img_h, const ncnn::Mat& in_pad,
                       const int& wpad, const int& hpad, std::vector<Object>& objects) {
    ncnn::Mat mask_pred_result;
    slice(output_mask, mask_pred_result, 0, in_pad.w - wpad, 2);
    slice(mask_pred_result, mask_pred_result, 0, in_pad.h - hpad, 1);
    interp(mask_pred_result, 1.0, img_w, img_h, mask_pred_result);

    std::vector<std::pair<float, int>> topk_indices;
    std::vector<int> labels_per_image;
    getTopk(output_class, 100, topk_indices, labels_per_image, mask_threshold);

    int picked = topk_indices.size();
    objects.resize(picked);
    int size = mask_pred_result.h * mask_pred_result.w;

    #pragma omp parallel for
    for (int i = 0; i < picked; i++) {

        float* ptr = mask_pred_result.channel(topk_indices[i].second);

        float sum = 0, sum2 = 0;
        for (int j = 0; j < size; j++) {
            if (ptr[j] > 0) {
                sum += sigmoid(ptr[j]) * ptr[j];
                sum2 += ptr[j];
            }
        }

        float score = topk_indices[i].first * (sum / (sum2 + 1e-6));

        cv::Mat cv_mask_32f = cv::Mat(cv::Size(img_w, img_h), CV_32FC1, mask_pred_result.channel(topk_indices[i].second));
        cv::Mat cv_mask_8u;
        cv_mask_32f.convertTo(cv_mask_8u, CV_8UC1, 255.0, 0);
        objects[i].cv_mask = cv_mask_8u;
        objects[i].prob = score;
        objects[i].label = labels_per_image[i];
    }
}

static int detect_mask2former(const cv::Mat& bgr, std::vector<Object>& objects) {
    const int target_size = 800;
    int img_w = bgr.cols;
    int img_h = bgr.rows;
    const float mask_threshold = 0.45f;

    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    int wpad = target_size - w; //(w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = target_size - h; //(h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT, 114.f);

    const float mean_vals[3] = {123.675, 116.280, 103.530};
    const float norm_vals[3] = {1 / 58.395, 1 / 57.120, 1 / 57.375};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    int level_index[3] = {0, (in_pad.h * in_pad.w) / (32 * 32), ((in_pad.h * in_pad.w) / (32 * 32) + (in_pad.h * in_pad.w) / (16 * 16))};
    ncnn::Mat level_start_index = ncnn::Mat(3, (void*)level_index);

    float spatial_shape[6] = {in_pad.h / 32.f, in_pad.w / 32.f, in_pad.h / 16.f, in_pad.w / 16.f, in_pad.h / 8.f, in_pad.w / 8.f};
    ncnn::Mat spatial_shapes = ncnn::Mat(6, (void*)spatial_shape).reshape(2, 3, 1);
    float spatial_shape_inv[6] = {in_pad.w / 32.f, in_pad.h / 32.f, in_pad.w / 16.f, in_pad.h / 16.f, in_pad.w / 8.f, in_pad.h / 8.f};
    ncnn::Mat offset_normalizer = ncnn::Mat(6, (void*)spatial_shape_inv).reshape(2, 3, 1, 1);

    std::vector<ncnn::Mat> reference_points(1);
    get_reference_points(spatial_shape, reference_points);

    ncnn::Mat pos_em1, pos_em2, pos_em3, pos1, pos2, pos3;
    position_embedding_layer(in_pad.w, in_pad.h, pos_em1, pos_em2, pos_em3, pos1, pos2, pos3);

    /////////////////////////////////////////////////////
    ncnn::Net backbone_net;
    backbone_net.load_param("../models/mask2former_r50_backbone.param");
    backbone_net.load_model("../models/mask2former_r50_backbone.bin");

    ncnn::Extractor extractor = backbone_net.create_extractor();
    extractor.input("images", in_pad);

    ncnn::Mat feature1, feature2, feature3, feature4;
    extractor.extract("301", feature1);
    extractor.extract("343", feature2);
    extractor.extract("405", feature3);
    extractor.extract("437", feature4);

    ////////////////////////////////////
    ncnn::Net pixel_decoder_net;
    pixel_decoder_net.register_custom_layer("ms_attn", ms_attn_layer_creator);
    pixel_decoder_net.load_param("../models/mask2former_pixel_decoder.param");
    pixel_decoder_net.load_model("../models/mask2former_pixel_decoder.bin");

    ncnn::Extractor extractor_pixel_decoder = pixel_decoder_net.create_extractor();

    extractor_pixel_decoder.input("pixel_decoder_features2", feature2);
    extractor_pixel_decoder.input("pixel_decoder_features3", feature3);
    extractor_pixel_decoder.input("pixel_decoder_features4", feature4);
    extractor_pixel_decoder.input("pixel_decoder_pos_em1", pos_em1);
    extractor_pixel_decoder.input("pixel_decoder_pos_em2", pos_em2);
    extractor_pixel_decoder.input("pixel_decoder_pos_em3", pos_em3);
    extractor_pixel_decoder.input("pixel_decoder_reference_points", reference_points[0]);
    extractor_pixel_decoder.input("pixel_decoder_spatial_shapes", spatial_shapes);
    extractor_pixel_decoder.input("pixel_decoder_offset_normalizer", offset_normalizer);
    extractor_pixel_decoder.input("mask_feature_res2", feature1);

    ncnn::Mat out, out1, out2, out3;
    extractor_pixel_decoder.extract("pixel_decoder_7121", out);
    slice(out, out1, 0, level_index[1], 0);
    slice(out, out2, level_index[1], level_index[2], 0);
    slice(out, out3, level_index[2], 2147483647, 0);
    transpose(out1, out1, 1);
    transpose(out2, out2, 1);
    transpose(out3, out3, 1);
    reshape(out3, out3, 256, 100, -1, 0);

    ncnn::Mat mask_features;
    extractor_pixel_decoder.input("mask_feature_317", out3);
    extractor_pixel_decoder.extract("mask_feature_363", mask_features); //mask_features
    ncnn::Mat mask_features_reshape = mask_features.reshape(mask_features.h * mask_features.w, mask_features.c, 1);

    //////////////////////////////////////////

    ncnn::Net mask_feature_decode_net;
    mask_feature_decode_net.register_custom_layer("gen_attn", gen_attn_layer_creator);
    mask_feature_decode_net.load_param("../models/mask2former_transformer_predictor.param");
    mask_feature_decode_net.load_model("../models/mask2former_transformer_predictor.bin");

    ncnn::Extractor extractor_mask_feature_decode = mask_feature_decode_net.create_extractor();
    extractor_mask_feature_decode.set_light_mode(false); // encounter crash bug when light mode is true
    extractor_mask_feature_decode.input("head0_mask_features", mask_features_reshape);
    extractor_mask_feature_decode.input("head1_pos1", pos1);
    extractor_mask_feature_decode.input("head2_pos2", pos2);
    extractor_mask_feature_decode.input("head3_pos3", pos3);
    extractor_mask_feature_decode.input("head1_out1", out1);
    extractor_mask_feature_decode.input("head2_out2", out2);
    extractor_mask_feature_decode.input("head3_out3", out3);

    ncnn::Mat output_class, output_mask;
    extractor_mask_feature_decode.extract("head9_20000", output_mask);  //output_mask
    extractor_mask_feature_decode.extract("head9_20001", output_class); //output_class
    ///////////////////////////////////////

    decode_ins(output_mask, output_class, mask_threshold, img_w, img_h, in_pad, wpad, hpad, objects);

    return 0;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_mask2former(m, objects);

    draw_objects(m, objects);

    return 0;
}
