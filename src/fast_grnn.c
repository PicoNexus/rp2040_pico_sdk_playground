#include "sha_rnn_intf.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "fastgrnn_rnn0_params.h"
#include "fastgrnn_rnn1_params.h"
#include "fastgrnn_fc_params.h"

#define EULER_NUMBER_F (2.71828182846f)

// clang-format off

const float INPUT_MEANS[32] = 
   {-2.20377017e+01, -2.09740796e+01, -2.00308448e+01, -1.97414773e+01, -1.95580346e+01, -1.93713805e+01, -1.92363509e+01, -1.89242205e+01,
    -1.89907052e+01, -1.90585489e+01, -1.90045962e+01, -1.89841197e+01, -1.90666160e+01, -1.90064234e+01, -1.89129300e+01, -1.89202742e+01,
    -1.88064987e+01, -1.87803075e+01, -1.87957479e+01, -1.86493413e+01, -1.86242518e+01, -1.85893956e+01, -1.84924049e+01, -1.85074954e+01,
    -1.85348288e+01, -1.85715110e+01, -1.86225900e+01, -1.86602925e+01, -1.87791281e+01, -1.89771674e+01, -1.92326400e+01, -1.98047760e+01};

const float INPUT_STDEVS[32] = 
   { 1.54209489e+00,  2.60126882e+00,  3.31368577e+00,  3.54765560e+00,  3.70311426e+00,  3.89125366e+00,  4.01894476e+00,  4.18691960e+00,
     4.12947707e+00,  4.03149124e+00,  3.98364006e+00,  3.92079621e+00,  3.82719223e+00,  3.78604063e+00,  3.77818002e+00,  3.73393584e+00,
     3.72222597e+00,  3.68757736e+00,  3.67426711e+00,  3.72325849e+00,  3.70055044e+00,  3.68152235e+00,  3.69910030e+00,  3.65209990e+00,
     3.57471336e+00,  3.51167803e+00,  3.45659565e+00,  3.39875794e+00,  3.34811364e+00,  3.30420072e+00,  3.24033154e+00,  2.94201732e+00};

// clang-format on

static inline float sigmoidf(float n)
{
    return (1 / (1 + powf(EULER_NUMBER_F, -n)));
}

static inline float expo(float y)
{
    if (y > 80)
        y = 80;
    return exp(y);
}

static float softmax(const float *xs, size_t n, size_t len)
{
    float sum = 0;
    for (size_t i = 0; i < len; i++)
        sum += expo(xs[i]);
    if (sum == 0)
        sum = 0.001;
    return (expo(xs[n])) / sum;
}

static void rnn0_process(const float input[32], const float hidden[64], float output[64])
{
    float z;
    float c;

    for (size_t i = 0; i < 32; i++)
    {
        for (size_t j = 0; j < 64; j++)
        {
            output[j] += GRNN0_W[j][i] * input[i];
        }
    }

    for (size_t j = 0; j < 64; j++)
    {
        for (size_t i = 0; i < 64; i++)
        {
            output[j] += GRNN0_U[j][i] * hidden[i];
        }
    }

    for (size_t j = 0; j < 64; j++)
    {
        z = output[j] + GRNN0_BIAS_GATE[j];
        z = sigmoidf(z);
        c = output[j] + GRNN0_BIAS_UPDATE[j];
        c = tanhf(c);

        output[j] = z * hidden[j] + (sigmoidf(GRNN0_ZETA) * (1.0 - z) + sigmoidf(GRNN0_NU)) * c;
    }
}

void sha_rnn_rnn0_process(const sha_rnn_input_t input, sha_rnn_rnn1_input_t output)
{
    float hidden[64] = {0.0f};

    for (size_t k = 0; k < SHARNN_BRICK_SIZE; k++)
    {
        memset(output, 0, sizeof(sha_rnn_rnn1_input_t));
        rnn0_process(input[k], hidden, output);
        memcpy(hidden, output, sizeof(hidden));
    }
}

static void rnn1_process(const float input[64], const float hidden[32], float output[32])
{
    float z;
    float c;

    for (size_t j = 0; j < 32; j++)
    {
        for (size_t i = 0; i < 64; i++)
        {
            output[j] += GRNN1_W[j][i] * input[i];
        }
    }

    for (size_t j = 0; j < 32; j++)
    {
        for (size_t i = 0; i < 32; i++)
        {
            output[j] += GRNN1_U[j][i] * hidden[i];
        }
    }

    for (size_t j = 0; j < 32; j++)
    {
        z = output[j] + GRNN1_BIAS_GATE[j];
        z = sigmoidf(z);
        c = output[j] + GRNN1_BIAS_UPDATE[j];
        c = tanhf(c);

        output[j] = z * hidden[j] + (sigmoidf(GRNN1_ZETA) * (1.0 - z) + sigmoidf(GRNN1_NU)) * c;
    }
}

void sha_rnn_rnn1_process(const sha_rnn_rnn1_input_t input, sha_rnn_fc_input_t output)
{
    static sha_rnn_rnn1_input_t rnn1_input_hist[9];
    static size_t rnn1_hist_idx = 0;

    float rnn1_hidden[32] = {0.0};

    memcpy(rnn1_input_hist[rnn1_hist_idx], input, sizeof(sha_rnn_rnn1_input_t));

    for (size_t i = 0; i < 9; i++)
    {
        size_t j = (rnn1_hist_idx + 1 + i) % 9;
        memset(output, 0, sizeof(sha_rnn_fc_input_t));
        rnn1_process(rnn1_input_hist[j], rnn1_hidden, output);
        memcpy(rnn1_hidden, output, sizeof(sha_rnn_fc_input_t));
    }

    rnn1_hist_idx++;

    if (rnn1_hist_idx == 9)
    {
        rnn1_hist_idx = 0;
    }
}

void sha_rnn_fc_process(const sha_rnn_fc_input_t input, sha_rnn_output_t output)
{
    memset(output, 0, 6 * sizeof(float));

    for (size_t j = 0; j < FC_OUT_DIM; j++)
    {
        for (size_t i = 0; i < FC_IN_DIM; i++)
        {
            output[j] += input[i] * FC_W[j][i];
        }
        output[j] += FC_B[j];
    }
}

void sha_rnn_get_max_prob(const sha_rnn_output_t input, float *max_prob, size_t *max_idx)
{
    float max_logit = input[0];
    *max_idx = 0;

    for (size_t j = 0; j < FC_OUT_DIM; j++)
    {
        if (input[j] > max_logit)
        {
            max_logit = input[j];
            *max_idx = j;
        }
    }

    *max_prob = softmax(input, *max_idx, FC_OUT_DIM);
}

void sha_rnn_process(const sha_rnn_input_t input, float *max_prob, size_t *max_idx)
{
    static float output[64] = {0.0f};
    static float output2[32] = {0.0f};
    static float output3[6] = {0.0f};

    sha_rnn_rnn0_process(input, output);
    sha_rnn_rnn1_process(output, output2);
    sha_rnn_fc_process(output2, output3);
    sha_rnn_get_max_prob(output3, max_prob, max_idx);
}

void sha_rnn_norm(float *input, size_t num)
{
    for (size_t i = 0; i < num; i++)
    {
        for (size_t j = 0; j < 32; j++)
        {
            input[i * 32 + j] = (input[i * 32 + j] - INPUT_MEANS[j]) / INPUT_STDEVS[j];
        }
    }
}
