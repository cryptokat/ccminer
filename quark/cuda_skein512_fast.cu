static cudaStream_t gpustream[MAX_GPUS] = { 0 };
static __constant__ uint64_t c_PaddedMessage80[2]; // padded message (80 bytes + padding)
__constant__ uint64_t precalcvalues[9];
static uint32_t *d_nonce[MAX_GPUS];

#define SWAB32(x)     cuda_swab32(x)

#define R(x, n)       ((x) >> (n))
#define Ch(x, y, z)   ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)  ((x & (y | z)) | (y & z))
#define S0(x)         (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define S1(x)         (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define s0(x)         (ROTR32(x, 7) ^ ROTR32(x, 18) ^ R(x, 3))
#define s1(x)         (ROTR32(x, 17) ^ ROTR32(x, 19) ^ R(x, 10))

__constant__ uint32_t sha256_endingTable[] = {
0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000200,
0x80000000, 0x01400000, 0x00205000, 0x00005088, 0x22000800, 0x22550014, 0x05089742, 0xa0000020,
0x5a880000, 0x005c9400, 0x0016d49d, 0xfa801f00, 0xd33225d0, 0x11675959, 0xf6e6bfda, 0xb30c1549,
0x08b2b050, 0x9d7c4c27, 0x0ce2a393, 0x88e6e1ea, 0xa52b4335, 0x67a16f49, 0xd732016f, 0x4eeb2e91,
0x5dbf55e5, 0x8eee2335, 0xe2bc5ec2, 0xa83f4394, 0x45ad78f7, 0x36f3d0cd, 0xd99c05e8, 0xb0511dc7,
0x69bc7ac4, 0xbd11375b, 0xe3ba71e5, 0x3b209ff2, 0x18feee17, 0xe25ad9e7, 0x13375046, 0x0515089d,
0x4f0d0f04, 0x2627484e, 0x310128d2, 0xc668b434, 0x420841cc, 0x62d311b8, 0xe59ba771, 0x85a7a484
};

__constant__ uint32_t sha256_constantTable[64] = {
0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};


#define TFBIG_KINIT(k0, k1, k2, k3, k4, k5, k6, k7, k8, t0, t1, t2) { \
		k8 = k0 ^ k1 ^ k2 ^ k3 ^ k4 ^ k5 ^ k6 ^ k7 ^ make_uint2(0xA9FC1A22UL, 0x1BD11BDA); \
		t2 = t0 ^ t1; \
	}

#define TFBIG_MIX8(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX(w0, w1, rc0); \
		TFBIG_MIX(w2, w3, rc1); \
		TFBIG_MIX(w4, w5, rc2); \
		TFBIG_MIX(w6, w7, rc3); \
	}

#define TFBIG_4e(s)  { \
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
		TFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
		TFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
		TFBIG_MIX8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
	}

#define TFBIG_4o(s)  { \
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
		TFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
		TFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
		TFBIG_MIX8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
	}

#define TFBIG_MIX_PRE(x0, x1, rc) { \
		x0 = x0 + x1; \
		x1 = ROTL64(x1, rc) ^ x0; \
				}

#define TFBIG_MIX8_UI2(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX_UI2(w0, w1, rc0); \
		TFBIG_MIX_UI2(w2, w3, rc1); \
		TFBIG_MIX_UI2(w4, w5, rc2); \
		TFBIG_MIX_UI2(w6, w7, rc3); \
		}

#define TFBIG_MIX8_PRE(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX_PRE(w0, w1, rc0); \
		TFBIG_MIX_PRE(w2, w3, rc1); \
		TFBIG_MIX_PRE(w4, w5, rc2); \
		TFBIG_MIX_PRE(w6, w7, rc3); \
				}

#define TFBIG_4e_UI2(s)  { \
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
		TFBIG_MIX8_UI2(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
		TFBIG_MIX8_UI2(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
		TFBIG_MIX8_UI2(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
		}

#define TFBIG_4e_PRE(s)  { \
		TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
		TFBIG_MIX8_PRE(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
		TFBIG_MIX8_PRE(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
		TFBIG_MIX8_PRE(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
				}

#define TFBIG_4o_UI2(s)  { \
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
		TFBIG_MIX8_UI2(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
		TFBIG_MIX8_UI2(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
		TFBIG_MIX8_UI2(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
		}

#define TFBIG_4o_PRE(s)  { \
		TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
		TFBIG_MIX8_PRE(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
		TFBIG_MIX8_PRE(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
		TFBIG_MIX8_PRE(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
		}


static void precalc(int thr_id, uint64_t *PaddedMessage)
{
    uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8;
    uint64_t t0, t1, t2;

    h0 = 0x4903ADFF749C51CEull;
    h1 = 0x0D95DE399746DF03ull;
    h2 = 0x8FD1934127C79BCEull;
    h3 = 0x9A255629FF352CB1ull;
    h4 = 0x5DB62599DF6CA7B0ull;
    h5 = 0xEABE394CA9D5C3F4ull;
    h6 = 0x991112C71A75B523ull;
    h7 = 0xAE18A40B660FCC33ull;
    h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ SPH_C64(0x1BD11BDAA9FC1A22);

    t0 = 64; // ptr
    t1 = 0x7000000000000000ull;
    t2 = 0x7000000000000040ull;

    uint64_t p[8];
    for (int i = 0; i<8; i++)
        p[i] = PaddedMessage[i];

    TFBIG_4e_PRE(0);
    TFBIG_4o_PRE(1);
    TFBIG_4e_PRE(2);
    TFBIG_4o_PRE(3);
    TFBIG_4e_PRE(4);
    TFBIG_4o_PRE(5);
    TFBIG_4e_PRE(6);
    TFBIG_4o_PRE(7);
    TFBIG_4e_PRE(8);
    TFBIG_4o_PRE(9);
    TFBIG_4e_PRE(10);
    TFBIG_4o_PRE(11);
    TFBIG_4e_PRE(12);
    TFBIG_4o_PRE(13);
    TFBIG_4e_PRE(14);
    TFBIG_4o_PRE(15);
    TFBIG_4e_PRE(16);
    TFBIG_4o_PRE(17);
    TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

    uint64_t buffer[9];

    buffer[0] = PaddedMessage[0] ^ p[0];
    buffer[1] = PaddedMessage[1] ^ p[1];
    buffer[2] = PaddedMessage[2] ^ p[2];
    buffer[3] = PaddedMessage[3] ^ p[3];
    buffer[4] = PaddedMessage[4] ^ p[4];
    buffer[5] = PaddedMessage[5] ^ p[5];
    buffer[6] = PaddedMessage[6] ^ p[6];
    buffer[7] = PaddedMessage[7] ^ p[7];
    buffer[8] = t2;
    CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(precalcvalues, buffer, sizeof(buffer), 0, cudaMemcpyHostToDevice, gpustream[thr_id]));
}