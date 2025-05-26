import torch.nn as nn
import torch
from modules.models_Denoiser import SuperResolution_x8_PixelShuffle

class ResnetBlock(nn.Module):
    """
        Define a Resnet block.
        Refer to "https://github.com/ermongroup/ncsn/blob/master/models/pix2pix.py"
    """

    def __init__(
        self, 
        dim,
        kernel_size=1, 
        padding_type='zero',
        norm_layer=nn.BatchNorm2d, 
        use_dropout=False, 
        use_bias=True, 
        act=None
    ):
        
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, kernel_size, padding_type,
            norm_layer, use_dropout, use_bias, act
        )

    def build_conv_block(
        self, 
        dim, 
        kernel_size, 
        padding_type, 
        norm_layer, 
        use_dropout, 
        use_bias, 
        act=nn.GELU()
    ):

        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer)
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 0
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=p, bias=use_bias)]
        if norm_layer:
            conv_block += [norm_layer(dim, momentum=0.1)]
        if act:
            conv_block += [act]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 0
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=p, bias=use_bias)]
        if norm_layer:
            conv_block += [norm_layer(dim, momentum=0.1)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections

        return out
    
    

def get_activation(act):
    if act.lower() == 'relu':
        func = nn.ReLU(inplace=True)
    elif act.lower() == 'gelu':
        func = nn.GELU()
    elif act.lower() == 'none':
        func = None
    else:
        raise NotImplementedError
    return func


class R2L(nn.Module):
    def __init__(
        self,

        input_dim = 4,
        D = 60,
        W = 256,
        output_dim = 3,
        profile = False,
        cnn_type = 'sr',
    ):
        super(R2L, self).__init__()
        #breakpoint()
        self.input_dim = input_dim
        self.profile = profile
        act = get_activation('gelu')
        D, W = D, W
        Ws = [W] * (D-1) + [3]
        self.head = nn.Sequential(
            *[nn.Conv2d(input_dim, Ws[0], 1), act]) if act else nn.Sequential(*[nn.Conv2d(input_dim, Ws[0], 1)]
        )
        #breakpoint()
        n_block = (D - 2) // 2 # 2 layers per resblock
        body = [ResnetBlock(W, act=act) for _ in range(n_block)]
        self.body = nn.Sequential(*body)
        
      
       
        # hard-coded up-sampling factors
        if cnn_type == 'sr':
            for i in range(100):
                print(cnn_type)
            n_conv = 2
            n_up_block = 3
            kernels = [64, 64, 16]

            up_blocks = []
            for i in range(n_up_block - 1):
                in_dim = Ws[-2] if not i else kernels[i]
                up_blocks += [nn.ConvTranspose2d(in_dim, kernels[i], 4, stride=2, padding=1)]
                up_blocks += [ResnetBlock(kernels[i], act=act) for _ in range(n_conv)]

            k, s, p = 4, 2, 1

            up_blocks += [nn.ConvTranspose2d(kernels[1], kernels[-1], k, stride=s, padding=p)]
            up_blocks += [ResnetBlock(kernels[-1], act=act)  for _ in range(n_conv)]
            up_blocks += [nn.Conv2d(kernels[-1], output_dim, 1), nn.Sigmoid()]
            self.tail = nn.Sequential(*up_blocks )
        elif cnn_type == 'sr_pixel_shuffle':
            for i in range(100):
                print(cnn_type)
            #breakpoint()
            self.tail = SuperResolution_x8_PixelShuffle(
                input_channels=Ws[-2],
                output_channels=output_dim,
                base_channels=64,
                activation='relu'
            ).cuda()

    
    def forward(self, x):
        
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)
        
        return x
        
        # return x
        #breakpoint()    
        # if self.profile:  # ?��로파?���? 모드?�� ?���? ?���? 측정
        #     if torch.cuda.is_available():
        #         # GPU ?��?��?�� ?�� ?��?��?�� 측정
        #         start_head = torch.cuda.Event(enable_timing=True)
        #         end_head = torch.cuda.Event(enable_timing=True)
        #         start_body = torch.cuda.Event(enable_timing=True)
        #         end_body = torch.cuda.Event(enable_timing=True)
        #         start_tail = torch.cuda.Event(enable_timing=True)
        #         end_tail = torch.cuda.Event(enable_timing=True)

        #         # Head ?���? 측정
        #         start_head.record()
        #         x = self.head(x)
        #         end_head.record()
                
        #         # Body ?���? 측정
        #         start_body.record()
        #         x = self.body(x) + x
        #         end_body.record()
                
        #         # Tail ?���? 측정
        #         start_tail.record()
        #         x = self.tail(x)
        #         end_tail.record()

        #         # CUDA ?��벤트 ?��기화
        #         torch.cuda.synchronize()

        #         # �?리초 ?��?���? ?���? 출력
        #         head_time = start_head.elapsed_time(end_head)
        #         body_time = start_body.elapsed_time(end_body)
        #         tail_time = start_tail.elapsed_time(end_tail)
                
        #         print(f"Head time: {head_time:.2f}ms")
        #         print(f"Body time: {body_time:.2f}ms")
        #         print(f"Tail time: {tail_time:.2f}ms")
        #         print(f"Total time: {(head_time + body_time + tail_time):.2f}ms")
        #         #breakpoint()
            
        #     else:
        #         # CPU ?��?��?��
        #         import time
                
        #         # Head ?���? 측정
        #         t0 = time.time()
        #         x = self.head(x)
        #         t1 = time.time()
                
        #         # Body ?���? 측정
        #         x = self.body(x) + x
        #         t2 = time.time()
                
        #         # Tail ?���? 측정
        #         x = self.tail(x)
        #         t3 = time.time()
                
        #         print(f"Head time: {(t1-t0)*1000:.2f}ms")
        #         print(f"Body time: {(t2-t1)*1000:.2f}ms")
        #         print(f"Tail time: {(t3-t2)*1000:.2f}ms")
        #         print(f"Total time: {(t3-t0)*1000:.2f}ms")
                
        
        # else:
        #     # ?��반적?�� 추론
        #     x = self.head(x)
        #     x = self.body(x) + x
        #     x = self.tail(x)
        
        # return x



import torch.nn as nn
import torch
from modules.models_Denoiser import SuperResolution_x8_PixelShuffle

class ResnetBlock(nn.Module):
    """
        Define a Resnet block.
        Refer to "https://github.com/ermongroup/ncsn/blob/master/models/pix2pix.py"
    """

    def __init__(
        self, 
        dim,
        kernel_size=1, 
        padding_type='zero',
        norm_layer=nn.BatchNorm2d, 
        use_dropout=False, 
        use_bias=True, 
        act=None
    ):
        
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, kernel_size, padding_type,
            norm_layer, use_dropout, use_bias, act
        )

    def build_conv_block(
        self, 
        dim, 
        kernel_size, 
        padding_type, 
        norm_layer, 
        use_dropout, 
        use_bias, 
        act=nn.GELU()
    ):

        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer)
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 0
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=p, bias=use_bias)]
        if norm_layer:
            conv_block += [norm_layer(dim, momentum=0.1)]
        if act:
            conv_block += [act]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 0
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=p, bias=use_bias)]
        if norm_layer:
            conv_block += [norm_layer(dim, momentum=0.1)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections

        return out
    
    

def get_activation(act):
    if act.lower() == 'relu':
        func = nn.ReLU(inplace=True)
    elif act.lower() == 'gelu':
        func = nn.GELU()
    elif act.lower() == 'none':
        func = None
    else:
        raise NotImplementedError
    return func


class R2L_body(nn.Module):
    def __init__(
        self,

        input_dim = 4,
        D = 60,
        W = 256,
        output_dim = 3,
        profile = False,
        cnn_type = 'sr',
        isTeacher = False,
    ):
        super(R2L_body, self).__init__()  # R2L -> R2L_body로 수정
        #breakpoint()
        self.input_dim = input_dim
        self.profile = profile
        act = get_activation('gelu')
        D, W = D, W
        Ws = [W] * (D-1) + [3]
        self.head = nn.Sequential(
            *[nn.Conv2d(input_dim, Ws[0], 1), act]) if act else nn.Sequential(*[nn.Conv2d(input_dim, Ws[0], 1)]
        )
        #breakpoint()
        n_block = (D - 2) // 2 # 2 layers per resblock
        body = [ResnetBlock(W, act=act) for _ in range(n_block)]
        self.body = nn.Sequential(*body)
        
        self.isTeacher = isTeacher
      
       
        # hard-coded up-sampling factors
        


        kernels = [64, 64, 16]

        up_blocks = []
        
        up_blocks += [nn.Conv2d(W, output_dim, 1), nn.Sigmoid()]
        self.tail = nn.Sequential(*up_blocks )
       

    
    def forward(self, x):
        
        x = self.head(x)
        x = self.body(x) + x
        if not self.isTeacher:
            x = self.tail(x)
            
        return x
        
        # return x
        #breakpoint()    
        # if self.profile:  # ?��로파?���? 모드?�� ?���? ?���? 측정
        #     if torch.cuda.is_available():
        #         # GPU ?��?��?�� ?�� ?��?��?�� 측정
        #         start_head = torch.cuda.Event(enable_timing=True)
        #         end_head = torch.cuda.Event(enable_timing=True)
        #         start_body = torch.cuda.Event(enable_timing=True)
        #         end_body = torch.cuda.Event(enable_timing=True)
        #         start_tail = torch.cuda.Event(enable_timing=True)
        #         end_tail = torch.cuda.Event(enable_timing=True)

        #         # Head ?���? 측정
        #         start_head.record()
        #         x = self.head(x)
        #         end_head.record()
                
        #         # Body ?���? 측정
        #         start_body.record()
        #         x = self.body(x) + x
        #         end_body.record()
                
        #         # Tail ?���? 측정
        #         start_tail.record()
        #         x = self.tail(x)
        #         end_tail.record()

        #         # CUDA ?��벤트 ?��기화
        #         torch.cuda.synchronize()

        #         # �?리초 ?��?���? ?���? 출력
        #         head_time = start_head.elapsed_time(end_head)
        #         body_time = start_body.elapsed_time(end_body)
        #         tail_time = start_tail.elapsed_time(end_tail)
                
        #         print(f"Head time: {head_time:.2f}ms")
        #         print(f"Body time: {body_time:.2f}ms")
        #         print(f"Tail time: {tail_time:.2f}ms")
        #         print(f"Total time: {(head_time + body_time + tail_time):.2f}ms")
        #         #breakpoint()
            
        #     else:
        #         # CPU ?��?��?��
        #         import time
                
        #         # Head ?���? 측정
        #         t0 = time.time()
        #         x = self.head(x)
        #         t1 = time.time()
                
        #         # Body ?���? 측정
        #         x = self.body(x) + x
        #         t2 = time.time()
                
        #         # Tail ?���? 측정
        #         x = self.tail(x)
        #         t3 = time.time()
                
        #         print(f"Head time: {(t1-t0)*1000:.2f}ms")
        #         print(f"Body time: {(t2-t1)*1000:.2f}ms")
        #         print(f"Tail time: {(t3-t2)*1000:.2f}ms")
        #         print(f"Total time: {(t3-t0)*1000:.2f}ms")
                
        
        # else:
        #     # ?��반적?�� 추론
        #     x = self.head(x)
        #     x = self.body(x) + x
        #     x = self.tail(x)
        
        # return x
