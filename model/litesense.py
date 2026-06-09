import torch
import torch.nn as nn
import torch.nn.functional as F

from .mobilenet.mobilenetv4 import MobileNetV4C

class PatchDepthEstimator(nn.Module):
    def __init__(self, zone_grid_rows=8, zone_grid_cols=8):
        super(PatchDepthEstimator, self).__init__()

        self.zone_rows = zone_grid_rows
        self.zone_cols = zone_grid_cols
        self.total_patches = zone_grid_rows * zone_grid_cols

        self.patch_encoder = PatchEncoder(inputs=4, features=[32, 64, 96, 960])
        self.cnh_encoder = CnhEncoder(inputs=18, features=[128, 256])
        
        self.patch_fusion_x2 = CnhPatchFusion(features=[32, 128], embedding_dim=128)
        self.patch_fusion_x4 = CnhPatchFusion(features=[64, 256], embedding_dim=256)

        self.patch_decoder = PatchDecoder(inputs=[128, 256, 96, 960], features=[32, 64, 128, 256])
         
        self._reset_parameters()
    
    def _reset_parameters(self):
        modules = [self.patch_encoder, self.cnh_encoder, self.patch_fusion_x4, self.patch_fusion_x2, self.patch_decoder]
        
        for s in modules:
            for m in s.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    

    def _fuse_in_patch(self, image, histogram, layer):
        # Patch internal Cross-Attention
        b, c, h, w = image.shape

        # Split image into zone_rows x zone_cols patches
        image = image.view(b, c, self.zone_rows, h//self.zone_rows,
                           self.zone_cols, w//self.zone_cols).permute(0, 2, 4, 1, 3, 5).contiguous()
        image = image.view(-1, c, h//self.zone_rows, w//self.zone_cols)                         # [B*M*N, C, H/M, W/N]

        b, c, _, _ = histogram.shape
        histogram = histogram.unsqueeze(-1).permute(0, 2, 1, 3, 4).contiguous()
        histogram = histogram.view(-1, c, 1, 1)                                                 # [B*M*N, C, 1, 1]

        fusion = layer(image, histogram)
        b, c, h, w = fusion.shape

        # Reassemble patches into complete image
        fusion = fusion.view(b//self.total_patches, self.zone_rows, self.zone_cols,
                            c, h, w).permute(0, 3, 1, 4, 2, 5).contiguous()
        fusion = fusion.view(b//self.total_patches, c, h*self.zone_rows, w*self.zone_cols)      # [B, C, H, W]

        return fusion

    
    def forward(self, rgb, tof, cnh):
        rgbd = torch.cat([rgb, tof], dim=1)
        patch_feats_x2, patch_feats_x4, patch_feats_x8, patch_feats_x16 = self.patch_encoder(rgbd)
        
        b, c, _, _ = cnh.shape
        cnh = cnh.view(b, c, -1)
        cnh_feats1, cnh_feats2 = self.cnh_encoder(cnh)
        
        # # Patch fusion of RGB-D & CNH.
        fusion_feats_x2 = self._fuse_in_patch(patch_feats_x2, cnh_feats1, self.patch_fusion_x2)
        fusion_feats_x4 = self._fuse_in_patch(patch_feats_x4, cnh_feats2, self.patch_fusion_x4)
        
        patch_pred = self.patch_decoder(patch_feats_x16, patch_feats_x8, fusion_feats_x4, fusion_feats_x2)
        
        return patch_pred
    

class PatchEncoder(nn.Module):
    def __init__(self, inputs=4, features=[64, 128, 256, 512]):
        super(PatchEncoder, self).__init__()
        
        self.encoder = MobileNetV4C("MNV4ConvSmall-Custom")
            
    def forward(self, x):
        
        feats_x2, feats_x4, feats_x8, feats_x16 = self.encoder(x)
        
        return feats_x2, feats_x4, feats_x8, feats_x16
    

class LinearConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(LinearConv, self).__init__()
        # 1D-Conv for MLP building.
        self.convs = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(output_channel, output_channel, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True))
    
    def forward(self, x):
        return self.convs(x)


class CnhEncoder(nn.Module):
    def __init__(self, inputs=18, features=[128, 256]):
        super(CnhEncoder, self).__init__()
   
        self.conv1 = LinearConv(inputs, features[0])
        self.conv2 = LinearConv(features[0], features[1])

    def forward(self, x):
        feat1 = self.conv1(x)                               # [B, 18, zones] -> [B, 128, zones]
        feat2 = self.conv2(feat1)                           # [B, 128, zones] -> [B, 256, zones]
        return feat1.unsqueeze(-1), feat2.unsqueeze(-1)     # [B, C, zones, 1]


class UpSampleBN(nn.Module):
    def __init__(self, input_channel, skip_channel, output_channel):
        super(UpSampleBN, self).__init__()
        # Concat features first, then upsample.
        self.up_convs = nn.Sequential(
            nn.Conv2d(input_channel, skip_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_channel),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(skip_channel, output_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.up_convs(x)


class PatchDecoder(nn.Module):
    def __init__(self, inputs=[64, 256, 256, 512], features=[32, 64, 128, 256]):
        super(PatchDecoder, self).__init__()
        
        self.up_sample1 = UpSampleBN(inputs[3], features[3], features[3])
        self.up_sample2 = UpSampleBN(inputs[2] + features[3], features[2], features[2])
        self.up_sample3 = UpSampleBN(inputs[1] + features[2], features[1], features[1])
        self.up_sample4 = UpSampleBN(inputs[0] + features[1], features[0], features[0])

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(features[0], 1, 3, padding=1),
            nn.Softplus()
        )

    def forward(self, x16, x8, x4, x2):

        # c4 * h/16 * w/16
        up1 = self.up_sample1(x16)
        # c3 * h/8 * w/8
        up2 = self.up_sample2(torch.cat((up1, x8), dim=1))
        # c2 * h/4 * w/4
        up3 = self.up_sample3(torch.cat((up2, x4), dim=1))
        # c1 * h/2 * w/2
        up4 = self.up_sample4(torch.cat((up3, x2), dim=1))
        # c * h * w
        pred = self.conv3x3(up4)
        # 1 * h * w

        return pred


class CnhPatchFusion(nn.Module):
    def __init__(self, features=[256, 256], embedding_dim=256):
        super(CnhPatchFusion, self).__init__()
        self.cross_attention = Attention(features[0], features[1], embedding_dim)

    def forward(self, patch_feats, cnh_feats):
        return self.cross_attention(query=patch_feats, key=cnh_feats, value=cnh_feats)
    

class Attention(nn.Module):
    def __init__(self, query_features, value_features, embedding_dim):
        super(Attention, self).__init__()
        
        self.query_conv = nn.Conv2d(query_features, embedding_dim, 1)
        self.key_conv = nn.Conv2d(value_features, embedding_dim, 1)
        self.value_conv = nn.Conv2d(value_features, embedding_dim, 1)
        self.scale = embedding_dim ** -0.5

    def forward(self, query, key, value):
        
        keys = self.key_conv(key).flatten(2)                                                    # (B, C, 1, 1) -> (B, C, 1)
        values = self.value_conv(value).flatten(2)                                              # (B, C, 1, 1) -> (B, C, 1)
        querys = self.query_conv(query)                                                         # (B, C, h, w)
        
        attention_scores = torch.matmul(querys.flatten(2).transpose(1, 2), keys) * self.scale   # [B*zones, h*w, 1]
        # Sigmoid gate instead of Softmax for single-key attention
        attention_weights = torch.sigmoid(attention_scores)                                 
        
        weighted_values = torch.matmul(attention_weights, values.transpose(1, 2))               # (B, h*w, 1) * (B, 1, C) = (B, h*w, C)
        weighted_values = weighted_values.transpose(1, 2).view_as(querys)                       # (B, C, h, w)
        
        return weighted_values


if __name__ == "__main__":
    # Run: python -m model.litesense
    from thop import profile

    model = PatchDepthEstimator()
    
    image = torch.randn(1, 3, 416, 416) 
    tof = torch.randn(1, 1, 416, 416) 
    cnh = torch.randn(1, 18, 8, 8)

    macs, params = profile(model, inputs=(image, tof, cnh), verbose=False)
        
    print(f"Params: {params / 1e6:.2f} M")
    print(f"FLOPs: {macs * 2 / 1e9:.2f} G")
