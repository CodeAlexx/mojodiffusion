# Exact SCAIL-2 14B inference contract.
#
# Evidence: zai-org/SCAIL-2, wan-scail2 commit
# 5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5,
# wan/configs/scail_config_14B.py and wan/modules/model_scail2.py.


@fieldwise_init
struct Scail2Config(Copyable, Movable, ImplicitlyCopyable):
    var num_layers: Int
    var dim: Int
    var ffn_dim: Int
    var num_heads: Int
    var head_dim: Int
    var in_dim: Int
    var mask_dim: Int
    var out_dim: Int
    var freq_dim: Int
    var text_dim: Int
    var text_len: Int
    var image_dim: Int
    var image_len: Int
    var patch_f: Int
    var patch_h: Int
    var patch_w: Int
    var eps: Float32
    var image_norm_eps: Float32
    var rope_theta: Float32

    @staticmethod
    def scail2_14b() -> Scail2Config:
        return Scail2Config(
            num_layers=40,
            dim=5120,
            ffn_dim=13824,
            num_heads=40,
            head_dim=128,
            in_dim=20,
            mask_dim=28,
            out_dim=16,
            freq_dim=256,
            text_dim=4096,
            text_len=512,
            image_dim=1280,
            image_len=257,
            patch_f=1,
            patch_h=2,
            patch_w=2,
            eps=1.0e-6,
            image_norm_eps=1.0e-5,
            rope_theta=10000.0,
        )

    def validate(self) raises:
        if self.dim != self.num_heads * self.head_dim:
            raise Error("SCAIL-2 hidden/head geometry mismatch")
        if self.head_dim != 128 or self.head_dim % 2 != 0:
            raise Error("SCAIL-2 head dimension must be the pinned even 128")
        if self.in_dim != 20 or self.mask_dim != 28 or self.out_dim != 16:
            raise Error("SCAIL-2 channel contract mismatch")
        if self.patch_f != 1 or self.patch_h != 2 or self.patch_w != 2:
            raise Error("SCAIL-2 patch contract mismatch")
        if self.text_len != 512 or self.text_dim != 4096:
            raise Error("SCAIL-2 text contract mismatch")
        if self.image_len != 257 or self.image_dim != 1280:
            raise Error("SCAIL-2 image-token contract mismatch")
