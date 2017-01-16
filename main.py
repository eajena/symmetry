import glob,sys,os,time
from skimage.draw import polygon, circle
from matplotlib.colors import hsv_to_rgb

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import caffe

def export_pdf(filename, figs=None, dpi=300):
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    if figs:
        font_size = plt.rcParams['font.size']
        plt.rcParams['font.size'] = 6
        pp = PdfPages(filename)
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        print len(figs), "figures written to", filename
        plt.rcParams['font.size'] = font_size
    else:
        print "No figures found!"

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0, norm=False):
    if norm:
        data -= data.min()
        data /= data.max()

    x,y = (6,16)
    # force the number of filters to be square
    #n = int(np.ceil(np.sqrt(data.shape[0])))
    #padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    #data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    print data.shape
    padding = ((0,0),(0,1),(0,1),(0,0))
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((x, y) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((x * data.shape[1], y * data.shape[3]) + data.shape[4:])
    return data


def get_file_list(directory, extensions, max_files=0):
    file_list = []
    for f in os.listdir(directory):
        name, file_ext = os.path.splitext(f)
        if file_ext in extensions:
            file_list.append(os.path.join(directory, name + file_ext))

    file_list = sorted(file_list)
    return file_list if max_files==0 else file_list[:max_files]


#-------------------------------------------------------------------------------

def draw_single_histogram(canvas, l, (mh,mw), size):
    minl, maxl = 0, size
    l = l + minl/float(maxl)
    n = float(l.shape[0])
    
    img = np.dstack([canvas]*3) if canvas.ndim<3 else canvas
    
    rr, cc = circle(mh, mw, minl+maxl)
    img[rr, cc, :] = (img[rr, cc, :]*2. + (0.8*255, 0.8*255, 0.8*255))/3.
    rr, cc = circle(mh, mw, minl)
    img[rr, cc, :] = (1.0*255, 1.0*255, 1.0*255)
    
    for a in range(int(n)):
        
        sx = np.array([mh+minl*np.cos((a-0.5)/n*2*np.pi), mh+minl*np.cos((a+0.5)/n*2*np.pi)])
        sy = np.array([mw+minl*np.sin((a-0.5)/n*2*np.pi), mw+minl*np.sin((a+0.5)/n*2*np.pi)])
    
        x = np.array([mh+maxl*l[a]*np.cos((a+0.5)/n*2*np.pi), mh+maxl*l[a]*np.cos((a-0.5)/n*2*np.pi)])
        y = np.array([mw+maxl*l[a]*np.sin((a+0.5)/n*2*np.pi), mw+maxl*l[a]*np.sin((a-0.5)/n*2*np.pi)])
        rr, cc = polygon(np.hstack((sx,x)), np.hstack((sy,y)))
    
        img[rr, cc, :] = tuple(i*255 for i in hsv_to_rgb((a/n,1,1)))

    #imsave(filename, img)
    return img

def draw_histograms(img, histogram):
    ph, pw, n = histogram.shape
    height, weight = img.shape

    result = img
    for h in range(ph):
        for w in range(pw):
            hist = histogram[h,w]
            result = draw_single_histogram(result, hist/np.max(histogram), (height/ph*(h+0.5), weight/pw*(w+0.5)), min(height/ph, weight/pw)/2-1)

    return result

#-------------------------------------------------------------------------------

def get_histogram(resp, patches=10, f=np.sum):
    (i_filters, ih, iw) = resp.shape
    histogram = np.zeros((patches,patches,i_filters))
    patch_h = ih/float(patches)
    patch_w = iw/float(patches)

    for h in range(patches):
        for w in range(patches):
            ph = h*patch_h
            pw = w*patch_w
            patch_val = resp[:,ph:ph+patch_h, pw:pw+patch_w]

            for b in range(i_filters):
                histogram[h,w,b] = f(patch_val[b])
    
    histogram_sum = np.sum(histogram, axis=2)
    normalized_histogram = histogram / histogram_sum[:,:,np.newaxis]

    return histogram, normalized_histogram

#-------------------------------------------------------------------------------

def get_symmetry(normalized_histogram_orig, normalized_histogram_flip):
    assert(normalized_histogram_orig.shape == normalized_histogram_flip.shape)

    sum_abs = np.sum(np.abs(normalized_histogram_orig - normalized_histogram_flip))
    sum_max = np.sum(np.maximum(normalized_histogram_orig, normalized_histogram_flip))
    return 1.0 - sum_abs / sum_max

#-------------------------------------------------------------------------------

def load_model(model):
    mean_file = os.path.join('Models/ilsvrc_2012_mean.npy')
    if model == "alexnet":
        model_weights = os.path.join('Models/bvlc_alexnet/bvlc_alexnet.caffemodel')
        model_model = os.path.join('Models/bvlc_alexnet/deploy.prototxt')
    elif model == "alexnet-600x800":
        model_weights = os.path.join('Models/bvlc_alexnet/bvlc_alexnet.caffemodel')
        model_model = os.path.join('Models/bvlc_alexnet/deploy-600x800.prototxt')
    elif model == "alexnet-512":
        model_weights = os.path.join('Models/bvlc_alexnet/bvlc_alexnet.caffemodel')
        model_model = os.path.join('Models/bvlc_alexnet/deploy-512.prototxt')
    elif model == "vgg":
        model_weights = os.path.join('Models/vgg/VGG_normalised.caffemodel')
        model_model = os.path.join('Models/vgg/VGG_ave_pool_deploy.prototxt')
    else:
        assert False, "Unknown model"

    caffe.set_mode_gpu()
    caffe.set_device(0)

    null_fds = os.open(os.devnull, os.O_RDWR)
    out_orig = os.dup(2)
    os.dup2(null_fds, 2)
    net = caffe.Net(model_model, model_weights, caffe.TEST)
    os.dup2(out_orig, 2)
    os.close(null_fds)

    transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
    transformer.set_mean("data", np.load(mean_file).mean(1).mean(1)) # imagenet mean (bgr)
    transformer.set_channel_swap("data", (2,1,0))
    transformer.set_transpose("data", (2,0,1))
    transformer.set_raw_scale("data", 255)

    return net, transformer

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    TIMESTAMP = time.strftime("%Y.%m.%d-%H%M%S-%a")
    BASE_DIR = "/home/ab/Documents/Laboratory/Images"

    MODEL = "alexnet-512"
    if len(sys.argv)!=3:
        layer,patches = "conv1",15
    else:
        layer,patches = sys.argv[1], int(sys.argv[2])
    EXP = "cover-symmetry-"+MODEL+"-"+layer+"-p"+str(patches)

    print "AE-Measures"
    print "  started", TIMESTAMP
    print "  base_dir =", BASE_DIR
    print "  model =", MODEL
    print "  exp =", EXP

    net, transformer = load_model(MODEL)

    # print network configuration
    print "Network configuration\n ",
    print "\n  ".join([str((k, v.data.shape)) for k, v in net.blobs.items()])

    CATEGORIES = ["Cover-Classic", "Cover-Metal", "Cover-Pop"]

    layers = [k for k, v in net.blobs.items()]

    if False:
        print "Showing visualization of filters"
        plt.imshow(vis_square(net.params[layer][0].data.copy().transpose(0, 2, 3, 1), norm=True), interpolation='none')
        plt.show()

    # /* clear log and write csv header */
    result_names = ["symmetry-lr", "symmetry-ud", "symmetry-lrud"]
    with open(EXP+'-'+TIMESTAMP+'.csv', 'w') as log:
        log.write("category,image,"+",".join(result_names)+"\n")
        log.flush()

        all_list_val = []
        for c in CATEGORIES:

            list_val = []
            files = get_file_list(os.path.join(BASE_DIR, c, 'images',), [".png",".tif",".jpg"], max_files=0)
            for f in files:
                
                img_name = os.path.join(BASE_DIR, c, 'images', f)
                img = caffe.io.load_image(img_name)

                print img_name
                l = layer
                vals = []

                source_img = transformer.preprocess('data',img)[None,:]
                net.forward(data = source_img)
                resp = net.blobs[l].data[0]
                histogram8_orig,normalized_histogram8_orig = get_histogram(resp, patches=patches, f=np.max)

                source_img = transformer.preprocess('data',np.fliplr(img))[None,:]
                net.forward(data = source_img)
                resp = net.blobs[l].data[0]
                histogram8_fliplr,normalized_histogram8_fliplr = get_histogram(resp, patches=patches, f=np.max)

                source_img = transformer.preprocess('data',np.flipud(img))[None,:]
                net.forward(data = source_img)
                resp = net.blobs[l].data[0]
                histogram8_flipud,normalized_histogram8_flipud = get_histogram(resp, patches=patches, f=np.max)

                source_img = transformer.preprocess('data',np.fliplr(np.flipud(img)))[None,:]
                net.forward(data = source_img)
                resp = net.blobs[l].data[0]
                histogram8_fliplrud,normalized_histogram8_fliplrud = get_histogram(resp, patches=patches, f=np.max)

                vals.append( get_symmetry(histogram8_orig, histogram8_fliplr))
                vals.append( get_symmetry(histogram8_orig, histogram8_flipud))
                vals.append( get_symmetry(histogram8_orig, histogram8_fliplrud))
                
                log.write(c+","+os.path.basename(img_name)+","+",".join([str(r) for r in vals])+"\n")
                print c+","+os.path.basename(img_name)+","+",".join([str(r) for r in vals])+"\n"
                
                if False:
                    plt.figure("histogram "+l),plt.imshow( draw_histograms(np.zeros((2000,2000),dtype='uint8'), histogram8_orig))
                    plt.figure("histogram-lr "+l),plt.imshow( draw_histograms(np.zeros((2000,2000),dtype='uint8'), histogram8_fliplr))
                    plt.figure("raw"), plt.title(f), plt.imshow(transformer.deprocess("data",source_img)), plt.colorbar()
                    plt.figure("filters-"+l), plt.title(l), plt.imshow(vis_square(resp, padval=0)), plt.colorbar()        
                    plt.show()

                list_val.append(vals)

            all_list_val.append(list_val)

    for idx,idx_name in enumerate(result_names):
        plt.figure(idx_name), plt.xticks(rotation=25), plt.title(idx_name), plt.boxplot( [[r[idx] for r in c] for c in all_list_val], labels=CATEGORIES)#, plt.ylim([0,1])
    export_pdf(EXP+"-"+TIMESTAMP+".pdf")
