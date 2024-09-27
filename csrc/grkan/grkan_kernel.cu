#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_LESS(x, y) TORCH_CHECK(x < y, #x " should be less than " #y)

namespace grkan {

    // Adjust as needed based on device capabilities
    const int threads_per_block = 1024;
    const int block_size = 512;

    template <typename scalar_t>
    __global__ void grkan_forward_cuda_kernel(
        const scalar_t* __restrict__ x, 
        const scalar_t* __restrict__ a,
        const scalar_t* __restrict__ b, 
        scalar_t* __restrict__ result, 
        int B, int L, int D, int group, 
        int x_size, int D_per_group) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= x_size) return;  // Prevent out-of-bounds memory access

        // Calculate the index within the dimension D
        int d_index = idx % D;
        // Calculate the group index based on the position within dimension D
        int g_index = floor(d_index / D_per_group);

        // Calculate specific indices for a and b based on group
        int a_idx = g_index * 6;
        int b_idx = g_index * 4;

        // Load coefficients into registers
        scalar_t s_a[6], s_b[4];
        for (int i = 0; i < 6; ++i) {
            s_a[i] = a[a_idx + i];
        }
        for (int i = 0; i < 4; ++i) {
            s_b[i] = abs(b[b_idx + i]);  // Store absolute values directly if needed
        }

        // Obtain the input value from the tensor
        scalar_t xp1 = x[idx];
        scalar_t abs_xp1 = abs(xp1);

        // Compute the polynomial for P using Horner's method
        scalar_t P = s_a[5];
        for (int i = 4; i >= 0; --i) {
            P = fmaf(P, xp1, s_a[i]);
        }
        
        // Compute the polynomial for Q using Horner's method
        scalar_t Q = s_b[3];
        for (int i = 2; i >= 0; --i) {
            Q = fmaf(Q, abs_xp1, s_b[i]);
        }
        Q = fmaf(Q, abs_xp1, 1.0);

        // Write the result of P / Q
        result[idx] = P / Q;
    }

    at::Tensor grkan_forward_cuda(at::Tensor x, at::Tensor n, at::Tensor d, int64_t group) {
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);
        CHECK_LESS(group, 32);
        
        auto result = at::empty_like(x);
        const int x_size = x.numel();
        int B = x.size(0);
        int L = x.size(1);
        int D = x.size(2);

        int num_blocks = (x_size + threads_per_block - 1) / threads_per_block;

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "grkan_forward_cuda", ([&] {
        grkan_forward_cuda_kernel<scalar_t>
            <<<num_blocks, threads_per_block>>>(
                x.data_ptr<scalar_t>(),
                n.data_ptr<scalar_t>(),
                d.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                B, L, D, group, x_size, D / group);
            }));

        return result;
    }


    template <typename scalar_t>
    __global__ void grkan_backward_cuda_kernel(
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ x,
        const scalar_t* __restrict__ a,
        const scalar_t* __restrict__ b,
        scalar_t* __restrict__ d_x,
        float* __restrict__ d_a,
        float* __restrict__ d_b,
        int B, int L, int D, int group, 
        int x_size, 
        const int n_size, 
        const int d_size,
        int D_per_group) {
        
        // Shared memory for accumulation
        // group < 32, so we can use 192 and 128 shared memory
        __shared__ float sda[192];
        __shared__ float sdb[128];
        // initialize shared memory to zero
        if ( threadIdx.x == 0) {
            for (int i = 0; i < 192; ++i) {
                sda[i] = 0;
            }
            for (int i = 0; i < 128; ++i) {
                sdb[i] = 0;
            }
        }

        __syncthreads();

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= x_size) return;  // Prevent out-of-bounds memory access

        // Calculate the index within the dimension D
        int d_index = idx % D;
        // Calculate the group index based on the position within dimension D
        int g_index = floor(d_index / D_per_group);

        // Calculate specific indices for a and b based on group
        int a_idx = g_index * 6;
        int b_idx = g_index * 4;

        // Load coefficients into registers
        scalar_t shared_a[6], shared_b_abs[4], shared_b[4];;
        for (int i = 0; i < 6; ++i) {
            shared_a[i] = a[a_idx + i];
        }
        for (int i = 0; i < 4; ++i) {
            shared_b_abs[i] = abs(b[b_idx + i]);  // Store absolute values directly if needed
            shared_b[i] = b[b_idx + i];
        }

        scalar_t local_da[6] = {0}; // Local accumulation arrays
        scalar_t local_db[4] = {0};
        
        scalar_t xp = x[idx];
        scalar_t axp = abs(xp);
        // Compute powers of xp
        scalar_t xp_powers[5];
        xp_powers[0] = xp;
        xp_powers[1] = xp * xp_powers[0]; // xp^2
        xp_powers[2] = xp * xp_powers[1]; // xp^3
        xp_powers[3] = xp * xp_powers[2]; // xp^4
        xp_powers[4] = xp * xp_powers[3]; // xp^5

        // Compute powers of axp
        scalar_t axp_powers[4];
        axp_powers[0] = axp;
        axp_powers[1] = axp * axp_powers[0]; // axp^2
        axp_powers[2] = axp * axp_powers[1]; // axp^3
        axp_powers[3] = axp * axp_powers[2]; // axp^4

        // Compute absolute values once

        scalar_t P = shared_a[0] 
        + shared_a[1] * xp_powers[0] 
        + shared_a[2] * xp_powers[1] 
        + shared_a[3] * xp_powers[2] 
        + shared_a[4] * xp_powers[3] 
        + shared_a[5] * xp_powers[4];

        scalar_t Q = scalar_t(1.0)
        + shared_b_abs[0] * axp_powers[0] 
        + shared_b_abs[1] * axp_powers[1] 
        + shared_b_abs[2] * axp_powers[2] 
        + shared_b_abs[3] * axp_powers[3];


        scalar_t R = shared_a[1] 
        + scalar_t(2.0) * shared_a[2] * xp_powers[0] 
        + scalar_t(3.0) * shared_a[3] * xp_powers[1] 
        + scalar_t(4.0) * shared_a[4] * xp_powers[2] 
        + scalar_t(5.0) * shared_a[5] * xp_powers[3];

        scalar_t S = copysign(scalar_t(1.0), xp) * (shared_b_abs[0] 
        + scalar_t(2.0) * shared_b_abs[1] * axp_powers[0] 
        + scalar_t(3.0) * shared_b_abs[2] * axp_powers[1] 
        + scalar_t(4.0) * shared_b_abs[3] * axp_powers[2]);
        

        scalar_t grad_o = grad_output[idx];
        
        scalar_t mpq2 = -P/(Q*Q);

        scalar_t d_i_x = (R / Q + S * mpq2) * grad_o;
        d_x[idx] = d_i_x;

        // Loop for computing d_a contributions
        local_da[0] = scalar_t(1.0) / Q * grad_o;
        for (int i = 1; i < 6; ++i) {
            local_da[i] = (xp_powers[i-1] / Q) * grad_o;
        }

        // Loop for computing d_b contributions
        for (int i = 0; i < 4; ++i) {
            local_db[i] = mpq2 * copysign(scalar_t(1.0), shared_b[i]) * axp_powers[i] * grad_o;
        }

        // Reduce local arrays to shared memory
        for (int i = 0; i < 6; ++i) {
            atomicAdd(&sda[a_idx + i], local_da[i]);
        }
        for (int i = 0; i < 4; ++i) {
            atomicAdd(&sdb[b_idx + i], local_db[i]);
        }

        __syncthreads();

        // Only one thread writes back to global memory
        if (threadIdx.x == 0) {
            for (int i = 0; i < n_size; ++i) {
                atomicAdd(&d_a[i], sda[i]);
            }
            for (int i = 0; i < d_size; ++i) {
                atomicAdd(&d_b[i], sdb[i]);
            }
        }
    }

    std::vector<at::Tensor> grkan_backward_cuda(at::Tensor grad_output, at::Tensor x, at::Tensor n, at::Tensor d, int64_t group) {
        CHECK_INPUT(grad_output);
        CHECK_INPUT(x);
        CHECK_INPUT(n);
        CHECK_INPUT(d);
        CHECK_LESS(group, 32);
    
        const int x_size = x.numel();
        const int n_size = n.numel();
        const int d_size = d.numel();

        auto d_x = at::empty_like(x);
        auto d_n = at::zeros_like(n).toType(at::kFloat);
        auto d_d = at::zeros_like(d).toType(at::kFloat);

        int B = x.size(0);
        int L = x.size(1);
        int D = x.size(2);

        int num_blocks = (x_size + block_size - 1) / block_size;

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "grkan_backward_cuda", ([&] {
        grkan_backward_cuda_kernel<scalar_t>
            <<<num_blocks, block_size>>>(
                grad_output.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                n.data_ptr<scalar_t>(),
                d.data_ptr<scalar_t>(),
                d_x.data_ptr<scalar_t>(),
                d_n.data_ptr<float>(),
                d_d.data_ptr<float>(),
                B, L, D, group, x_size, n_size, d_size, D / group);
        }));

        return {d_x, d_n, d_d};
    }

}
