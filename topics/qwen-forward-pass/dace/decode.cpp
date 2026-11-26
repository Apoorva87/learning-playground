/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct qwen_block3_decode_state_t {

};

void __program_qwen_block3_decode_internal(qwen_block3_decode_state_t*__state, float * __restrict__ denom, float * __restrict__ exps, float * __restrict__ k, float * __restrict__ out, float * __restrict__ peak, float * __restrict__ probs, float * __restrict__ q, float * __restrict__ scores, float * __restrict__ v)
{

    {

        {
            for (auto h = 0; h < 24; h += 1) {
                for (auto t = 0; t < 17; t += 1) {
                    {
                        float* keys = &k[((256 * t) + (4352 * (h / 6)))];
                        float* query = &q[(256 * h)];
                        float result;

                        ///////////////////
                        float acc = 0.0f;
                        for (int j = 0; j < 256; ++j) acc += query[j] * keys[j];
                        result = acc * 0.0625f;
                        ///////////////////

                        scores[((17 * h) + t)] = result;
                    }
                }
            }
        }

    }
    {

        {
            for (auto h = 0; h < 24; h += 1) {
                {
                    float* row = &scores[(17 * h)];
                    float result;

                    ///////////////////
                    float maximum = row[0];
                    for (int j = 1; j < 17; ++j) maximum = std::max(maximum, row[j]);
                    result = maximum;
                    ///////////////////

                    peak[h] = result;
                }
            }
        }

    }
    {

        {
            for (auto h = 0; h < 24; h += 1) {
                for (auto t = 0; t < 17; t += 1) {
                    {
                        float maximum = peak[h];
                        float score = scores[((17 * h) + t)];
                        float result;

                        ///////////////////
                        result = std::exp(score - maximum);
                        ///////////////////

                        exps[((17 * h) + t)] = result;
                    }
                }
            }
        }

    }
    {

        {
            for (auto h = 0; h < 24; h += 1) {
                {
                    float* row = &exps[(17 * h)];
                    float result;

                    ///////////////////
                    float acc = 0.0f;
                    for (int j = 0; j < 17; ++j) acc += row[j];
                    result = acc;
                    ///////////////////

                    denom[h] = result;
                }
            }
        }

    }
    {

        {
            for (auto h = 0; h < 24; h += 1) {
                for (auto t = 0; t < 17; t += 1) {
                    {
                        float total = denom[h];
                        float weight = exps[((17 * h) + t)];
                        float result;

                        ///////////////////
                        result = weight / total;
                        ///////////////////

                        probs[((17 * h) + t)] = result;
                    }
                }
            }
        }

    }
    {

        {
            for (auto h = 0; h < 24; h += 1) {
                for (auto d = 0; d < 256; d += 1) {
                    {
                        float* weights = &probs[(17 * h)];
                        float* values = &v[(d + (4352 * (h / 6)))];
                        float result;

                        ///////////////////
                        float acc = 0.0f;
                        for (int j = 0; j < 17; ++j) acc += weights[j] * values[j * 256];
                        result = acc;
                        ///////////////////

                        out[(d + (256 * h))] = result;
                    }
                }
            }
        }

    }
}

DACE_EXPORTED void __program_qwen_block3_decode(qwen_block3_decode_state_t *__state, float * __restrict__ denom, float * __restrict__ exps, float * __restrict__ k, float * __restrict__ out, float * __restrict__ peak, float * __restrict__ probs, float * __restrict__ q, float * __restrict__ scores, float * __restrict__ v)
{
    __program_qwen_block3_decode_internal(__state, denom, exps, k, out, peak, probs, q, scores, v);
}

DACE_EXPORTED qwen_block3_decode_state_t *__dace_init_qwen_block3_decode()
{
    int __result = 0;
    qwen_block3_decode_state_t *__state = new qwen_block3_decode_state_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED int __dace_exit_qwen_block3_decode(qwen_block3_decode_state_t *__state)
{
    int __err = 0;
    delete __state;
    return __err;
}
