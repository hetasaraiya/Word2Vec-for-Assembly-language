from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label
from bokeh.models.tools import HoverTool
from bokeh.plotting import figure, show, output_file
from sklearn.manifold import TSNE
import gensim


def plot_with_pca():
    labels = []
    tokens = []
    for word in model.wv.vocab:
        if (word.startswith("rex")):
            continue
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    vector = ['valign', 'vp', 'vblend', 'vbroadcast', 'vcompress', 'vcvt', 'vdbs', 'vexpand', 'vextract', 'vfixup',
              'vfmadd', 'vfmsub', 'vfnmadd', 'vfnmsub', 'vfpclass', 'vgather', 'vget', 'vinsert', 'vmask', 'vmov',
              'vpblend',
              'vpbroadcast', 'vpcmp', 'vpcompress', 'vpconflict', 'vperm', 'vpexpand', 'vpgather', 'vplzcnt', 'vpmadd',
              'vpmask', 'vpmov', 'vmult', 'vpro', 'vpscatter', 'vps', 'vpt', 'vran', 'vrcp', 'vred', 'vrnd', 'vrsq',
              'vsca', 'vshuf',
              'vtest', 'vzero', 'vector']
    read = ['rdpid', 'rdpkru', 'rdpmc', 'rdrand', 'rdseed', 'rdtsc']
    mask = ['kadd', 'kand', 'kor', 'kmov', 'knot', 'kshift', 'ktest', 'kunpck', 'kxnor', 'kxor']
    bound = ['bndcl', 'bndcn', 'bndcu', 'bndldx', 'bndmk', 'bndmov', 'bndstx']
    string = ['cmps', 'cmpsb', 'cmpsq', 'cmpsw', 'lods', 'movs', 'rep', 'scas', 'stos']
    flag = ['clac', 'clc', 'cld', 'cli', 'clts', 'cmc', 'lahf', 'popf', 'pushf', 'sahf', 'stac', 'stc', 'std']
    segment_reg = ['lds', 'les', 'lfs', 'lgs', 'lss', 'rdfs', 'rdgs', 'rdmsr', 'swapgs', 'verr', 'verw', 'wrfs', 'wrgs',
                   'xgetbv', 'xsetbv']
    misc = ['cpuid', 'lea', 'nop', 'xlat']
    floating_control = ['fclex', 'fcmov', 'fdecstp', 'ffree', 'fincstp', 'finit', 'fldcw', 'fldenv', 'fnclex', 'fninit',
                        'fnop', 'fnsave', 'fnstcw', 'fnstenv', 'fnstsw', 'frstor', 'fsave', 'fstcw', 'fstenv', 'fstsw',
                        'fwait', 'fxrstor', 'fxsave']
    floating_data = ['fbld', 'fbstp', 'fild', 'fist', 'fld', 'fst', 'fxch']
    floating_arith = ['fabs', 'fadd', 'faddp', 'fchs', 'fdiv', 'fiadd', 'fidiv', 'fimul', 'fisub', 'fmul', 'frndint',
                      'fscale', 'fsub']
    floating_comparison = ['fcom', 'ficom', 'ftst', 'fucom', 'fxam']
    floating_transcendental = ['f2xmi', 'fcos', 'fpatan', 'fprem', 'fptan', 'fsin', 'fsqrt', 'fxtract', 'fyl2x']
    simd_state = ['fxrstor', 'fxsave']
    mmx = ['packssdw', 'packsswb', 'packusdw', 'packuswb',
           'pabs', 'padd', 'palignr', 'pblend', 'pclmulqdql', 'phadd', 'phmin', 'phsub', 'pmadd', 'psadbw',
           'psign', 'package', 'pcmp', 'pand', 'por', 'ptest', 'pxor', 'psra', 'psrl', 'emms', 'xabort', 'xacquire',
           'xbegin', 'xrelease', 'xrstor', 'xsave', 'xtest', 'xend']
    sse = ['extractps', 'movaps', 'movups', 'addps', 'addss', 'addsubps', 'blendps', 'blendvps', 'divps', 'divss',
           'dpps', 'haddps', 'hsubps',
           'maxps', 'maxss', 'minps', 'minss', 'mulps', 'mulss', 'sqrtps', 'sqrtss', 'subps', 'subss',
           'cmpps', 'cmpss', 'comiss', 'rcpps', 'rcpss', 'ucomiss', 'andnps', 'andps', 'orps', 'xorps', 'shufps',
           'unpckhps', 'unpckhlps',
           'cvtdq2ps', 'cvtdpd2ps', 'cvtpi2ps', 'cvtps2dq', 'cvtps2pd', 'cvtps2pi', 'cvtss2sd', 'cvtss2si',
           'cvttps2dq', 'cvttps2pi', 'movhlps', 'movhps', 'movlhps', 'movlps', 'movmskps', 'rsqrt', 'maskmovq',
           'movntps',
           'movntq', 'pause', 'prefetch', 'sfence']
    sse2 = ['movapd', 'movddup', 'movupd', 'addpd', 'addsd', 'addsubpd', 'blendpd', 'blendvpd', 'divpd', 'divsd',
            'dppd', 'haddpd', 'hsubpd',
            'maxpd', 'maxsd', 'minpd', 'minsd', 'mpsadbw', 'mulpd', 'mulsd', 'sqrtpd', 'sqrtsd', 'subpd', 'subsd',
            'cmppd', 'cmpsd', 'comisd', 'ucomisd', 'andnpd', 'andpd', 'orpd', 'xorpd', 'shufpd', 'unpckhpd',
            'unpckhlpd', 'cvtdq2pd', 'cvtdpd2dq', 'cvtdpd2pi', 'cvtpi2pd', 'cvtsd2si', 'cvtsd2ss', 'cvtsi2ss',
            'cvtsi2sd',
            'cvttpd2dq', 'cvttpd2pi', 'cvttsd2si', 'cvttss2si', 'movdq2q', 'movdqa', 'movdqu', 'movhpd', 'movlpd',
            'movmskpd', 'movq2dq', 'pavg', 'pextr', 'pinsr',
            'pmax', 'pmin', 'pmov', 'pmul', 'pshuf', 'psll', 'psub', 'punpck', 'roundpd', 'roundps', 'roundsd',
            'roundss', 'clflush', 'clflushopt', 'clwb', 'lfence',
            'maskmovdqu', 'mfence', 'movntdq', 'movnti', 'movntpd']
    system = ['hlt', 'invd', 'invlpg', 'invpcid', 'lgdt', 'lidt', 'lldt', 'lmsw', 'lock', 'lsl', 'ltr', 'rsm', 'sgdt',
              'sidt', 'sldt', 'smsw', 'stmxcsr', 'str', 'wbinvd', 'wrmsr', 'wrpkru', 'ldmxcsr']
    encryption = ['aesdec', 'aesdeclast', 'aesenc', 'aesenclast', 'aesimc', 'aeskeygenassist', 'sha1', 'sha256']
    mmx_data = ['movd', 'movq']
    dec_arith = ['aaa', 'aad', 'aam', 'aas', 'adcx', 'daa', 'das']
    binary_arith = ['adc', 'add', 'adox', 'cmp', 'dec', 'div', 'idiv', 'imul', 'inc', 'mul', 'neg', 'sbb', 'sub',
                    'tzcnt']
    logical = ['and', 'andn', 'not', 'or', 'xor']
    shift = ['rcl', 'rcr', 'rol', 'ror', 'sal', 'sar', 'shl', 'shr']
    bit_byte = ['bextr', 'blsi', 'blsmsk', 'blsr', 'bsf', 'bsr', 'bt', 'btc', 'btr', 'bts', 'lzcnt', 'set', 'test']
    control_transfer = ['bound', 'call', 'enter', 'int', 'iret', 'jmp', 'leave', 'loop', 'ret', 'wait', 'sysret']
    data_transfer = ['bswap', 'bzhi', 'cbw', 'cdq', 'cdqe', 'cmov', 'cmpxchg', 'cqo', 'crc32', 'cwd', 'cwde', 'lar',
                     'lddqu', 'mov', 'pdep', 'pext', 'pop', 'ptwrite', 'push', 'xadd']
    io = ['in', 'ins', 'out']
    color = []
    type1 = []
    for i in range(len(labels)):
        c = '#000000'
        t = ''
        if (labels[i].lower().startswith(tuple(vector)) > 0):
            c = '#800000'
            t = 'vector'
        elif (labels[i].lower().startswith(tuple(read)) > 0):
            c = '#CD853F'
            t = 'read'
        elif (labels[i].lower().startswith(tuple(mask)) > 0):
            c = '#D2691E'
            t = 'mask'
        elif (labels[i].lower().startswith(tuple(bound)) > 0):
            c = '#DAA520'
            t = 'bound'
        elif (labels[i].lower().startswith(tuple(string)) > 0):
            c = '#CD5C5C'
            t = 'string'
        elif (labels[i].lower().startswith(tuple(flag)) > 0):
            c = '#FA8072'
            t = 'flag'
        elif (labels[i].lower().startswith(tuple(segment_reg)) > 0):
            c = '#DC143C'
            t = 'segment register'
        elif (labels[i].lower().startswith(tuple(misc)) > 0):
            c = '#FF0000'
            t = 'misc'
        elif (labels[i].lower().startswith(tuple(floating_control)) > 0):
            c = '#8B0000'
            t = 'floating control'
        elif (labels[i].lower().startswith(tuple(floating_data)) > 0):
            c = '#FFC0CB'
            t = 'floating data'
        elif (labels[i].lower().startswith(tuple(floating_arith)) > 0):
            c = '#FF69B4'
            t = 'floating arith'
        elif (labels[i].lower().startswith(tuple(floating_comparison)) > 0):
            c = '#FF1493'
            t = 'floating comparison'
        elif (labels[i].lower().startswith(tuple(floating_transcendental)) > 0):
            c = '#DB7093'
            t = 'floating transcendental'
        elif (labels[i].lower().startswith(tuple(simd_state)) > 0):
            c = '#FF7F50'
            t = 'simd state'
        elif (labels[i].lower().startswith(tuple(mmx)) > 0):
            c = '#FF4500'
            t = 'mmx'
        elif (labels[i].lower().startswith(tuple(sse)) > 0):
            c = '#FFFF00'
            t = 'sse'
        elif (labels[i].lower().startswith(tuple(sse2)) > 0):
            c = '#FFE4B5'
            t = 'sse2'
        elif (labels[i].lower().startswith(tuple(system)) > 0):
            c = '#F0E68C'
            t = 'system'
        elif (labels[i].lower().startswith(tuple(encryption)) > 0):
            c = '#E6E6FA'
            t = 'encryption'
        elif (labels[i].lower().startswith(tuple(mmx_data)) > 0):
            c = '#DDA0DD'
            t = 'mmx_data'
        elif (labels[i].lower().startswith(tuple(dec_arith)) > 0):
            c = '#EE82EE'
            t = 'dec_arith'
        elif (labels[i].lower().startswith(tuple(binary_arith)) > 0):
            c = '#4B0082'
            t = 'binary arith'
        elif (labels[i].lower().startswith(tuple(logical)) > 0):
            c = '#008000'
            t = 'logical'
        elif (labels[i].lower().startswith(tuple(shift)) > 0):
            c = '#808000'
            t = 'shift'
        elif (labels[i].lower().startswith(tuple(bit_byte)) > 0):
            c = '#00FFFF'
        elif (labels[i].lower().startswith(tuple(control_transfer)) > 0):
            c = '#00BFFF'
            t = 'bit byte'
        elif (labels[i].lower().startswith(tuple(data_transfer)) > 0):
            c = '#0000FF'
            t = 'data transfer'
        elif (labels[i].lower().startswith(tuple(io)) > 0):
            c = '#00FF00'
            t = 'IO'
        color.append(c)
        type1.append(t)

    output_file("Models/word2vec_some_pca.html", title="word2vec_out example")

    source = ColumnDataSource(
        data=dict(
            x=x,
            y=y,
            desc=labels,
            typ=type1,
            fill_color=color
        )
    )
    hover = HoverTool(
        tooltips=[

            ("desc", "@desc"),
            ("typ", "@typ")
        ]
    )

    p = figure(plot_width=1500, plot_height=1000, tools=[hover],
               title="Word2Vec Plot With PCA (3 Iter)")

    p.circle('x', 'y', size=10, source=source, fill_color="fill_color")
    labels1 = LabelSet(x='x', y='y', text='desc', level='glyph',
                       x_offset=5, y_offset=5, source=source, render_mode='canvas')
    p.add_layout(labels1)
    show(p)


def plot_without_pca():
    labels = []
    tokens = []

    for word in model.wv.vocab:
        if (word.startswith("rex")):
            continue

        tokens.append(model[word])
        labels.append(word)
    tsne_model = TSNE(n_components=2)
    X_tsne = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in X_tsne:
        x.append(value[0])
        y.append(value[1])
    vector = ['valign', 'vp', 'vblend', 'vbroadcast', 'vcompress', 'vcvt', 'vdbs', 'vexpand', 'vextract', 'vfixup',
              'vfmadd', 'vfmsub', 'vfnmadd', 'vfnmsub', 'vfpclass', 'vgather', 'vget', 'vinsert', 'vmask', 'vmov',
              'vpblend',
              'vpbroadcast', 'vpcmp', 'vpcompress', 'vpconflict', 'vperm', 'vpexpand', 'vpgather', 'vplzcnt', 'vpmadd',
              'vpmask', 'vpmov', 'vmult', 'vpro', 'vpscatter', 'vps', 'vpt', 'vran', 'vrcp', 'vred', 'vrnd', 'vrsq',
              'vsca', 'vshuf',
              'vtest', 'vzero', 'vector']
    read = ['rdpid', 'rdpkru', 'rdpmc', 'rdrand', 'rdseed', 'rdtsc']
    mask = ['kadd', 'kand', 'kor', 'kmov', 'knot', 'kshift', 'ktest', 'kunpck', 'kxnor', 'kxor']
    bound = ['bndcl', 'bndcn', 'bndcu', 'bndldx', 'bndmk', 'bndmov', 'bndstx']
    string = ['cmps', 'cmpsb', 'cmpsq', 'cmpsw', 'lods', 'movs', 'rep', 'scas', 'stos']
    flag = ['clac', 'clc', 'cld', 'cli', 'clts', 'cmc', 'lahf', 'popf', 'pushf', 'sahf', 'stac', 'stc', 'std']
    segment_reg = ['lds', 'les', 'lfs', 'lgs', 'lss', 'rdfs', 'rdgs', 'rdmsr', 'swapgs', 'verr', 'verw', 'wrfs', 'wrgs',
                   'xgetbv', 'xsetbv']
    misc = ['cpuid', 'lea', 'nop', 'xlat']
    floating_control = ['fclex', 'fcmov', 'fdecstp', 'ffree', 'fincstp', 'finit', 'fldcw', 'fldenv', 'fnclex', 'fninit',
                        'fnop', 'fnsave', 'fnstcw', 'fnstenv', 'fnstsw', 'frstor', 'fsave', 'fstcw', 'fstenv', 'fstsw',
                        'fwait', 'fxrstor', 'fxsave']
    floating_data = ['fbld', 'fbstp', 'fild', 'fist', 'fld', 'fst', 'fxch']
    floating_arith = ['fabs', 'fadd', 'faddp', 'fchs', 'fdiv', 'fiadd', 'fidiv', 'fimul', 'fisub', 'fmul', 'frndint',
                      'fscale', 'fsub']
    floating_comparison = ['fcom', 'ficom', 'ftst', 'fucom', 'fxam']
    floating_transcendental = ['f2xmi', 'fcos', 'fpatan', 'fprem', 'fptan', 'fsin', 'fsqrt', 'fxtract', 'fyl2x']
    simd_state = ['fxrstor', 'fxsave']
    mmx = ['packssdw', 'packsswb', 'packusdw', 'packuswb',
           'pabs', 'padd', 'palignr', 'pblend', 'pclmulqdql', 'phadd', 'phmin', 'phsub', 'pmadd', 'psadbw',
           'psign', 'package', 'pcmp', 'pand', 'por', 'ptest', 'pxor', 'psra', 'psrl', 'emms', 'xabort', 'xacquire',
           'xbegin', 'xrelease', 'xrstor', 'xsave', 'xtest', 'xend']
    sse = ['extractps', 'movaps', 'movups', 'addps', 'addss', 'addsubps', 'blendps', 'blendvps', 'divps', 'divss',
           'dpps', 'haddps', 'hsubps',
           'maxps', 'maxss', 'minps', 'minss', 'mulps', 'mulss', 'sqrtps', 'sqrtss', 'subps', 'subss',
           'cmpps', 'cmpss', 'comiss', 'rcpps', 'rcpss', 'ucomiss', 'andnps', 'andps', 'orps', 'xorps', 'shufps',
           'unpckhps', 'unpckhlps',
           'cvtdq2ps', 'cvtdpd2ps', 'cvtpi2ps', 'cvtps2dq', 'cvtps2pd', 'cvtps2pi', 'cvtss2sd', 'cvtss2si',
           'cvttps2dq', 'cvttps2pi', 'movhlps', 'movhps', 'movlhps', 'movlps', 'movmskps', 'rsqrt', 'maskmovq',
           'movntps',
           'movntq', 'pause', 'prefetch', 'sfence']
    sse2 = ['movapd', 'movddup', 'movupd', 'addpd', 'addsd', 'addsubpd', 'blendpd', 'blendvpd', 'divpd', 'divsd',
            'dppd', 'haddpd', 'hsubpd',
            'maxpd', 'maxsd', 'minpd', 'minsd', 'mpsadbw', 'mulpd', 'mulsd', 'sqrtpd', 'sqrtsd', 'subpd', 'subsd',
            'cmppd', 'cmpsd', 'comisd', 'ucomisd', 'andnpd', 'andpd', 'orpd', 'xorpd', 'shufpd', 'unpckhpd',
            'unpckhlpd', 'cvtdq2pd', 'cvtdpd2dq', 'cvtdpd2pi', 'cvtpi2pd', 'cvtsd2si', 'cvtsd2ss', 'cvtsi2ss',
            'cvtsi2sd',
            'cvttpd2dq', 'cvttpd2pi', 'cvttsd2si', 'cvttss2si', 'movdq2q', 'movdqa', 'movdqu', 'movhpd', 'movlpd',
            'movmskpd', 'movq2dq', 'pavg', 'pextr', 'pinsr',
            'pmax', 'pmin', 'pmov', 'pmul', 'pshuf', 'psll', 'psub', 'punpck', 'roundpd', 'roundps', 'roundsd',
            'roundss', 'clflush', 'clflushopt', 'clwb', 'lfence',
            'maskmovdqu', 'mfence', 'movntdq', 'movnti', 'movntpd']
    system = ['hlt', 'invd', 'invlpg', 'invpcid', 'lgdt', 'lidt', 'lldt', 'lmsw', 'lock', 'lsl', 'ltr', 'rsm', 'sgdt',
              'sidt', 'sldt', 'smsw', 'stmxcsr', 'str', 'wbinvd', 'wrmsr', 'wrpkru', 'ldmxcsr']
    encryption = ['aesdec', 'aesdeclast', 'aesenc', 'aesenclast', 'aesimc', 'aeskeygenassist', 'sha1', 'sha256']
    mmx_data = ['movd', 'movq']
    dec_arith = ['aaa', 'aad', 'aam', 'aas', 'adcx', 'daa', 'das']
    binary_arith = ['adc', 'add', 'adox', 'cmp', 'dec', 'div', 'idiv', 'imul', 'inc', 'mul', 'neg', 'sbb', 'sub',
                    'tzcnt']
    logical = ['and', 'andn', 'not', 'or', 'xor']
    shift = ['rcl', 'rcr', 'rol', 'ror', 'sal', 'sar', 'shl', 'shr']
    bit_byte = ['bextr', 'blsi', 'blsmsk', 'blsr', 'bsf', 'bsr', 'bt', 'btc', 'btr', 'bts', 'lzcnt', 'set', 'test']
    control_transfer = ['bound', 'call', 'enter', 'int', 'iret', 'jmp', 'leave', 'loop', 'ret', 'wait', 'sysret']
    data_transfer = ['bswap', 'bzhi', 'cbw', 'cdq', 'cdqe', 'cmov', 'cmpxchg', 'cqo', 'crc32', 'cwd', 'cwde', 'lar',
                     'lddqu', 'mov', 'pdep', 'pext', 'pop', 'ptwrite', 'push', 'xadd']
    io = ['in', 'ins', 'out']
    color = []
    type1 = []
    for i in range(len(labels)):
        c = '#000000'
        t = ''
        if (labels[i].lower().startswith(tuple(vector)) > 0):
            c = '#800000'
            t = 'vector'
        elif (labels[i].lower().startswith(tuple(read)) > 0):
            c = '#CD853F'
            t = 'read'
        elif (labels[i].lower().startswith(tuple(mask)) > 0):
            c = '#D2691E'
            t = 'mask'
        elif (labels[i].lower().startswith(tuple(bound)) > 0):
            c = '#DAA520'
            t = 'bound'
        elif (labels[i].lower().startswith(tuple(string)) > 0):
            c = '#CD5C5C'
            t = 'string'
        elif (labels[i].lower().startswith(tuple(flag)) > 0):
            c = '#FA8072'
            t = 'flag'
        elif (labels[i].lower().startswith(tuple(segment_reg)) > 0):
            c = '#DC143C'
            t = 'segment register'
        elif (labels[i].lower().startswith(tuple(misc)) > 0):
            c = '#FF0000'
            t = 'misc'
        elif (labels[i].lower().startswith(tuple(floating_control)) > 0):
            c = '#8B0000'
            t = 'floating control'
        elif (labels[i].lower().startswith(tuple(floating_data)) > 0):
            c = '#FFC0CB'
            t = 'floating data'
        elif (labels[i].lower().startswith(tuple(floating_arith)) > 0):
            c = '#FF69B4'
            t = 'floating arith'
        elif (labels[i].lower().startswith(tuple(floating_comparison)) > 0):
            c = '#FF1493'
            t = 'floating comparison'
        elif (labels[i].lower().startswith(tuple(floating_transcendental)) > 0):
            c = '#DB7093'
            t = 'floating transcendental'
        elif (labels[i].lower().startswith(tuple(simd_state)) > 0):
            c = '#FF7F50'
            t = 'simd state'
        elif (labels[i].lower().startswith(tuple(mmx)) > 0):
            c = '#FF4500'
            t = 'mmx'
        elif (labels[i].lower().startswith(tuple(sse)) > 0):
            c = '#FFFF00'
            t = 'sse'
        elif (labels[i].lower().startswith(tuple(sse2)) > 0):
            c = '#FFE4B5'
            t = 'sse2'
        elif (labels[i].lower().startswith(tuple(system)) > 0):
            c = '#F0E68C'
            t = 'system'
        elif (labels[i].lower().startswith(tuple(encryption)) > 0):
            c = '#E6E6FA'
            t = 'encryption'
        elif (labels[i].lower().startswith(tuple(mmx_data)) > 0):
            c = '#DDA0DD'
            t = 'mmx_data'
        elif (labels[i].lower().startswith(tuple(dec_arith)) > 0):
            c = '#EE82EE'
            t = 'dec_arith'
        elif (labels[i].lower().startswith(tuple(binary_arith)) > 0):
            c = '#4B0082'
            t = 'binary arith'
        elif (labels[i].lower().startswith(tuple(logical)) > 0):
            c = '#008000'
            t = 'logical'
        elif (labels[i].lower().startswith(tuple(shift)) > 0):
            c = '#808000'
            t = 'shift'
        elif (labels[i].lower().startswith(tuple(bit_byte)) > 0):
            c = '#00FFFF'
        elif (labels[i].lower().startswith(tuple(control_transfer)) > 0):
            c = '#00BFFF'
            t = 'bit byte'
        elif (labels[i].lower().startswith(tuple(data_transfer)) > 0):
            c = '#0000FF'
            t = 'data transfer'
        elif (labels[i].lower().startswith(tuple(io)) > 0):
            c = '#00FF00'
            t = 'IO'
        color.append(c)
        type1.append(t)

    output_file("Models/word2vec_some_wpca.html", title="word2vec_out example")

    source = ColumnDataSource(
        data=dict(
            x=x,
            y=y,
            desc=labels,
            typ=type1,
            fill_color=color
        )
    )
    hover = HoverTool(
        tooltips=[

            ("desc", "@desc"),
            ("typ", "@typ")
        ]
    )

    p = figure(plot_width=1500, plot_height=1000, tools=[hover],
               title="Word2Vec Plot Without PCA (3 Iter)")

    p.circle('x', 'y', size=10, source=source, fill_color="fill_color")
    labels1 = LabelSet(x='x', y='y', text='desc', level='glyph',
                       x_offset=5, y_offset=5, source=source, render_mode='canvas')
    p.add_layout(labels1)
    show(p)


model = gensim.models.Word2Vec.load("Models/model_some.out")
plot_without_pca()
# plot_with_pca()

