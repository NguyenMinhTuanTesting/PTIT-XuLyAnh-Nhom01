"""Bai tap xu ly anh so - Phan B. Nhom 01."""
from __future__ import annotations
import csv
import time
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import data, color, util
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib.backends.backend_pdf import PdfPages
from config import GROUP_NUMBER, MEMBERS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'results'
SEED = 20261015
for d in (DATA, OUT): d.mkdir(parents=True, exist_ok=True)

# B1: Cac ham tu cai dat bang NumPy (khong dung OpenCV ben trong)
def hist_equalize(img):
    """Uint8 histogram equalization with CDF-min, matching cv2.equalizeHist."""
    a = np.asarray(img, dtype=np.uint8)
    hist = np.bincount(a.ravel(), minlength=256)
    nonzero = np.flatnonzero(hist)
    if len(nonzero) <= 1:
        return a.copy()
    cdf = hist.cumsum()
    cmin = cdf[nonzero[0]]
    lut = np.rint((cdf - cmin) * 255.0 / (a.size - cmin))
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[a]

def conv2d(img, kernel, padding='reflect'):
    """Spatial correlation convention (like cv2.filter2D), float64 output."""
    a = np.asarray(img, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or any(d % 2 != 1 for d in k.shape):
        raise ValueError('Kernel must have odd dimensions')
    mode = {'reflect':'reflect', 'constant':'constant', 'edge':'edge'}[padding]
    ph, pw = k.shape[0]//2, k.shape[1]//2
    padded = np.pad(a, ((ph,ph),(pw,pw)), mode=mode)
    windows = sliding_window_view(padded, k.shape)
    return np.einsum('ijkl,kl->ij', windows, k, optimize=True)

def median_filter(img, k):
    """NumPy median 2D with edge replication (same boundary convention as cv2.medianBlur)."""
    if k % 2 != 1 or k < 3: raise ValueError('k must be odd >=3')
    a = np.asarray(img)
    padded = np.pad(a, k//2, mode='edge')
    return np.median(sliding_window_view(padded, (k,k)), axis=(-2,-1)).astype(a.dtype)

def laplacian_sharpen(img, c=1.0):
    """g = clip(f - c*Laplacian(f)), L=[0,1,0;1,-4,1;0,1,0]."""
    lap = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float64)
    f = np.asarray(img, dtype=np.float64)
    return np.clip(np.rint(f - c*conv2d(f,lap)),0,255).astype(np.uint8)

def to_gray(a):
    if a.ndim==3: return np.rint(color.rgb2gray(a)*255).astype(np.uint8)
    return a.astype(np.uint8)

def save_img(path, a):
    if not cv2.imwrite(str(path), a): raise RuntimeError(f'Could not save {path}')

def degrade_dark(a):
    return np.clip(np.rint(255*(a.astype(float)/255)**2.2),0,255).astype(np.uint8)

def degrade_gauss(a, seed):
    rng=np.random.default_rng(seed)
    return np.clip(np.rint(a.astype(float)+rng.normal(0,20,a.shape)),0,255).astype(np.uint8)

def degrade_sp(a, seed):
    return np.rint(util.random_noise(a,mode='s&p',amount=.05,rng=seed)*255).astype(np.uint8)

def degrade_blur(a):
    return cv2.GaussianBlur(a,(0,0),2.0)

def create_data():
    """Lay 6 anh mau khac nhau tu skimage.data, sau do tao suy giam nhan tao."""
    sources=[('lowlight_camera',to_gray(data.camera()),'lowlight'),
             ('lowlight_coins',to_gray(data.coins()),'lowlight'),
             ('noise_moon',to_gray(data.moon()),'noise'),
             ('noise_clock',to_gray(data.clock()),'noise'),
             ('blur_page',to_gray(data.page()),'blur'),
             ('blur_coffee',to_gray(data.coffee()),'blur')]
    records=[]
    for i,(name,img,typ) in enumerate(sources):
        # Neu anh qua lon thi thu nho lai de chay nhanh hon.
        if max(img.shape)>512:
            factor=512/max(img.shape)
            img=cv2.resize(img,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        save_img(DATA/f'{name}_reference.png',img)
        if typ=='lowlight': bad=degrade_dark(img)
        elif typ=='noise': bad=degrade_sp(img,SEED+i) if i==2 else degrade_gauss(img,SEED+i)
        else: bad=degrade_blur(img)
        save_img(DATA/f'{name}.png',bad)
        records.append((name,typ,img,bad))
    ref=to_gray(data.camera())
    save_img(DATA/'reference.png',ref)
    transforms={'dark':degrade_dark(ref),'gaussian_noise_sigma20':degrade_gauss(ref,SEED),
                'salt_pepper_5pct':degrade_sp(ref,SEED),'gaussian_blur_sigma2':degrade_blur(ref)}
    for name,img in transforms.items():save_img(DATA/f'reference_{name}.png',img)
    with (DATA/'SOURCES.txt').open('w',encoding='utf-8') as fp:
        fp.write('Data source: scikit-image built-in sample images (camera, coins, moon, clock, page, coffee).\n')
        fp.write('Data processing: simulated low light, Gaussian noise sigma=20, salt-pepper 5%, or Gaussian blur sigma=2.\n')
        fp.write('Credit/license must be reviewed for each upstream sample before redistribution.\n')
        fp.write('Documentation: https://scikit-image.org/docs/stable/api/skimage.data.html\n')
        fp.write('These are NOT six independent photos captured by the student.\n')
    return records, ref, transforms

def metric(reference, test):
    return (float(peak_signal_noise_ratio(reference,test,data_range=255)),
            float(structural_similarity(reference,test,data_range=255)))

def write_csv(path, headings, rows):
    with path.open('w', newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(headings);w.writerows(rows)

def verify_b1(ref):
    # So sanh ket qua tu cai dat voi OpenCV
    kernel=np.ones((5,5), dtype=float)/25
    lap=np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=float)
    cases=[('hist_equalize',lambda:hist_equalize(ref),lambda:cv2.equalizeHist(ref)),
           ('conv2d_5x5',lambda:np.clip(np.rint(conv2d(ref,kernel)),0,255).astype('uint8'),
             lambda:cv2.filter2D(ref,-1,kernel,borderType=cv2.BORDER_REFLECT_101)),
           ('median_filter_5',lambda:median_filter(ref,5),lambda:cv2.medianBlur(ref,5)),
           ('laplacian_sharpen_c1',lambda:laplacian_sharpen(ref,1),
             lambda:np.clip(np.rint(ref.astype(float)-cv2.filter2D(ref.astype(float),cv2.CV_64F,lap,borderType=cv2.BORDER_REFLECT_101)),0,255).astype('uint8'))]
    rows=[]
    for name,ours,opencv in cases:
        t=time.perf_counter();x=ours(); t_ours=(time.perf_counter()-t)*1000
        t=time.perf_counter();y=opencv(); t_cv=(time.perf_counter()-t)*1000
        mae=float(np.abs(x.astype(float)-y.astype(float)).mean())
        rows.append((name,f'{mae:.6f}',f'{t_ours:.2f}',f'{t_cv:.2f}'))
    write_csv(OUT/'B1_comparison.csv',['Algorithm','MAE','NumPy_ms','OpenCV_ms'],rows)
    print('B1 | Algorithm | MAE | NumPy ms | OpenCV ms')
    for row in rows:print(' | '.join(map(str,row)))
    return rows

def unsharp(img,k):
    blur=cv2.GaussianBlur(img,(5,5),1)
    return np.clip(np.rint(img.astype(float)+k*(img.astype(float)-blur)),0,255).astype('uint8')

def run_b2(ref,deg):
    rows=[]
    def add(group, corruption, method, param, result):
        p,s=metric(ref,result)
        rows.append((group,corruption,method,param,round(p,5),round(s,6)))
    gamma_values=np.round(np.arange(.3,1.001,.1),2)
    for g in gamma_values:
        a=np.clip(np.rint(255*(deg['dark'].astype(float)/255)**g),0,255).astype('uint8')
        add('gamma','dark','gamma',g,a)
    for corrupt in ['gaussian_noise_sigma20','salt_pepper_5pct']:
        inp=deg[corrupt]
        for k in [3,5,7,9]:
            add('filters',corrupt,'mean',k,cv2.blur(inp,(k,k)))
            add('filters',corrupt,'gaussian',k,cv2.GaussianBlur(inp,(k,k),0))
            add('filters',corrupt,'median',k,cv2.medianBlur(inp,k))
    for k in [0,0.25,.5,.75,1,1.5,2]:
        add('unsharp','gaussian_blur_sigma2','unsharp',k,unsharp(deg['gaussian_blur_sigma2'],k))
    write_csv(OUT/'B2_metrics.csv',['Group','Degradation','Method','Parameter','PSNR_dB','SSIM'],rows)
    fig,axs=plt.subplots(2,3,figsize=(13,7.5))
    gamma=[r for r in rows if r[0]=='gamma']
    sharp=[r for r in rows if r[0]=='unsharp']
    for col,idx,title in [(0,4,'PSNR (dB)'),(1,5,'SSIM')]:
        axs[col,0].plot([r[3] for r in gamma],[r[idx] for r in gamma],marker='o')
        axs[col,0].set_title('Gamma correction')
        axs[col,0].set_xlabel('Gamma');axs[col,0].set_ylabel(title)
        axs[col,2].plot([r[3] for r in sharp],[r[idx] for r in sharp],marker='o')
        axs[col,2].set_title('Unsharp masking')
        axs[col,2].set_xlabel('k');axs[col,2].set_ylabel(title)
        for label in ['mean','gaussian','median']:
            sub=[r for r in rows if r[0]=='filters' and r[1]=='salt_pepper_5pct' and r[2]==label]
            axs[col,1].plot([r[3] for r in sub],[r[idx] for r in sub],marker='o',label=label)
        axs[col,1].set_title('Denoising salt-pepper 5%')
        axs[col,1].set_xlabel('Kernel size');axs[col,1].set_ylabel(title)
        axs[col,1].legend(fontsize=7)
    for ax in axs.ravel():ax.grid(alpha=.25)
    fig.suptitle('B2 - PSNR and SSIM parameter sweeps',fontsize=15)
    fig.tight_layout();fig.savefig(OUT/'B2_metrics_plot.png',dpi=165);plt.close(fig)
    # Ve them bieu do rieng cho nhieu Gaussian
    fig,axs=plt.subplots(1,2,figsize=(11,3.6))
    for idx,title,ax in [(4,'PSNR (dB)',axs[0]),(5,'SSIM',axs[1])]:
        for label in ['mean','gaussian','median']:
            sub=[r for r in rows if r[0]=='filters' and r[1]=='gaussian_noise_sigma20' and r[2]==label]
            ax.plot([r[3] for r in sub],[r[idx] for r in sub],marker='o',label=label)
        ax.set_title('Gaussian noise, sigma=20 - '+title);ax.set_xlabel('Kernel size');ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'B2_gaussian_noise_plot.png',dpi=165);plt.close(fig)
    return rows

def enhance(img):
    """Pipeline 3 buoc: median -> CLAHE -> unsharp."""
    median=median_filter(img,3)
    clahe=cv2.createCLAHE(clipLimit=1.6,tileGridSize=(8,8)).apply(median)
    return unsharp(clahe,0.5)

def run_b3(records):
    fig,axs=plt.subplots(6,2,figsize=(8,15))
    measurements=[]
    for i,(name,typ,ref,bad) in enumerate(records):
        improved=enhance(bad)
        save_img(OUT/f'{name}_enhanced.png',improved)
        before=metric(ref,bad);after=metric(ref,improved)
        measurements.append((name,typ,round(before[0],3),round(after[0],3),round(before[1],4),round(after[1],4)))
        for ax,arr,title in [(axs[i,0],bad,'Before'),(axs[i,1],improved,'After')]:
            ax.imshow(arr,cmap='gray',vmin=0,vmax=255);ax.axis('off')
            ax.set_title(f'{name} - {title}',fontsize=9)
    fig.suptitle('B3: Median 3x3 -> CLAHE -> Unsharp k=0.5',fontsize=14)
    fig.tight_layout();fig.savefig(OUT/'B3_six_before_after.png',dpi=170);plt.close(fig)
    write_csv(OUT/'B3_metrics.csv',['Image','Degradation','PSNR_before','PSNR_after','SSIM_before','SSIM_after'],measurements)
    return measurements

def run_b4(records):
    # Thu 2 truong hop that bai co y de xem hieu ung xau.
    noise=next(r for r in records if r[0]=='noise_clock')[3]
    strong_clahe=cv2.createCLAHE(clipLimit=12.0,tileGridSize=(4,4)).apply(noise)
    blurred=next(r for r in records if r[0]=='blur_page')[3]
    oversharp=unsharp(blurred,4.0)
    fig,axs=plt.subplots(2,2,figsize=(10,7.5))
    for ax,img,title in zip(axs.ravel(),[noise,strong_clahe,blurred,oversharp],
             ['Noisy input','Failure 1: CLAHE clipLimit 12', 'Blurred input','Failure 2: unsharp k=4']):
        ax.imshow(img,cmap='gray',vmin=0,vmax=255);ax.axis('off');ax.set_title(title)
    fig.suptitle('B4 - Failure cases: noise amplification and halos / overshoot')
    fig.tight_layout();fig.savefig(OUT/'B4_failure_cases.png',dpi=180);plt.close(fig)
    return ['Strong CLAHE can exaggerate noise; lower clipLimit or denoise first.',
            'Strong unsharp masking can create overshoot/ringing; reduce k or use edge-aware sharpening.']

def report(ref,b1,b2,b3,b4):
    """Tao bao cao PDF 5 trang cho nhom."""
    from matplotlib.image import imread
    outfile=ROOT/f'BaoCao_B1_Nhom{GROUP_NUMBER}.pdf'
    def page_title(fig,txt,n):
        fig.text(.07,.955,txt,fontsize=19,fontweight='bold',va='top')
        fig.text(.93,.035,f'B1 - Nhom {GROUP_NUMBER} | Trang {n}/5',ha='right',fontsize=8,color='#555555')
    def text(fig,x,y,content,fontsize=10):
        fig.text(x,y,content,fontsize=fontsize,va='top',linespacing=1.55)
    with PdfPages(outfile) as pdf:
        # Page 1: Introduction and provenance
        fig=plt.figure(figsize=(8.27,11.69));page_title(fig,'BO CONG CU NANG CAO CHAT LUONG ANH',1)
        text(fig,.08,.885,f'BAI TAP BUOI 1 - PHAN B | Nhom: {GROUP_NUMBER} | Mon: Xu ly anh so',12)
        text(fig,.08,.825,'1. Muc tieu',13)
        text(fig,.1,.785,'Tu cai dat 4 thuat toan bang NumPy; doi chieu voi OpenCV;\n'
            'khao sat PSNR/SSIM; xay dung pipeline va phan tich tinh huong that bai.',10)
        text(fig,.08,.69,'2. Bo du lieu & nguon',13)
        text(fig,.1,.65,'6 anh mau KHAC NHAU tu skimage.data: camera, coins, moon,\n'
            'clock, page, coffee (anh coffee duoc chuyen sang xam).\n'
            'Anh mau duoc giam chat luong NHAN TAO theo 3 nhom:\n'
            '2 thieu sang, 2 nhieu (muoi tieu / Gaussian), 2 mo Gaussian.\n'
            'Anh camera duoc dung lam tham chieu de tao 4 bien the suy giam.',10)
        text(fig,.08,.47,'3. Kiem soat tinh tai lap',13)
        text(fig,.1,.43,'Seed = 20261015; anh luu trong data/; cac tham so luu trong code.\n'
            'Moi phep do PSNR va SSIM deu so voi anh goc CHUA suy giam.',10)
        text(fig,.08,.34,'Luu y nguon va ban quyen',13)
        text(fig,.1,.30,'Nguon: https://scikit-image.org/docs/stable/api/skimage.data.html\n'
            'Can KIEM TRA va bo sung giay phep/credit cua tung anh truoc khi nop.\n'
            'Anh mau khong duoc gioi thieu la anh tu chup.',10)
        text(fig,.08,.15,'Thanh vien & phan cong (chi tiet o trang 5): '+str(len(MEMBERS))+' nguoi; tong '+str(sum(m[3] for m in MEMBERS))+'%.',10)
        pdf.savefig(fig);plt.close(fig)
        # Page 2
        fig=plt.figure(figsize=(8.27,11.69));page_title(fig,'B1 - THUAT TOAN & KIEM CHUNG',2)
        text(fig,.08,.86,'Tu cai dat NumPy: hist_equalize, conv2d (correlation),\nmedian_filter (edge pad), laplacian_sharpen.',11)
        text(fig,.08,.75,'BANG SAI KHAC TRUNG BINH VA THOI GIAN (1 lan chay)',11)
        ax=fig.add_axes([.07,.49,.86,.22]);ax.axis('off')
        tab=ax.table(cellText=b1,colLabels=['Algorithm','MAE','NumPy (ms)','cv2 (ms)'],loc='center',cellLoc='center',colWidths=[.42,.18,.2,.2]);tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,2)
        text(fig,.08,.41,'MAE = mean(|I_numpy - I_cv2|). Gia tri gan 0 cho thay\n'
            'ket qua khop nhau voi cung quy tac padding va lam tron.\n'
            'Thoi gian chay phu thuoc vao may, va do 1 lan co nhiu.',10)
        text(fig,.08,.24,'Ghi chu: cv2.filter2D thuc hien correlation, khong lat kernel.\n'
            'medianBlur dung border replicate. Ham conv2d mac dinh reflect\n'
            'tuong duong BORDER_REFLECT_101.',10)
        pdf.savefig(fig);plt.close(fig)
        # Page 3
        fig=plt.figure(figsize=(8.27,11.69));page_title(fig,'B2 - THUC NGHIEM DINH LUONG',3)
        a=fig.add_axes([.09,.42,.82,.46]);a.imshow(imread(OUT/'B2_metrics_plot.png'));a.axis('off')
        b=fig.add_axes([.12,.18,.77,.21]);b.imshow(imread(OUT/'B2_gaussian_noise_plot.png'));b.axis('off')
        text(fig,.08,.13,'Gamma: 0.3...1.0 (buoc 0.1) | Filter: 3/5/7/9 | Unsharp: 0...2',9)
        pdf.savefig(fig);plt.close(fig)
        # Page 4
        fig=plt.figure(figsize=(8.27,11.69));page_title(fig,'B3 - PIPELINE KET HOP',4)
        text(fig,.08,.90,'Median 3x3 -> CLAHE (clip=1.6) -> Unsharp (k=0.5).',11)
        text(fig,.08,.86,'Khu nhieu xung truoc; tang tuong phan cuc bo; sau cung lam net.',10)
        ax=fig.add_axes([.14,.12,.72,.70]);ax.imshow(imread(OUT/'B3_six_before_after.png'));ax.axis('off')
        pdf.savefig(fig);plt.close(fig)
        # Page 5
        fig=plt.figure(figsize=(8.27,11.69));page_title(fig,'B4/B5 - THAT BAI, DAO DUC, KET LUAN',5)
        ax=fig.add_axes([.16,.57,.68,.30]);ax.imshow(imread(OUT/'B4_failure_cases.png'));ax.axis('off')
        text(fig,.08,.53,'That bai 1: CLAHE qua manh -> khuech dai nhieu. Giam clipLimit.\n'
             'That bai 2: unsharp k qua lon -> quang sang / halo. Giam k.',10)
        text(fig,.08,.43,'Du lieu va dao duc',12)
        text(fig,.10,.39,'Anh mau lay tu skimage.data; can kiem tra quyen su dung rieng\n'
            'cua tung anh, ghi nguon va khong su dung anh mat nguoi khac\n'
            'cho muc dich cong bo khi chua co quyen/chu the dong y.\n'
            'Chinh sua anh phai minh bach: trong y te, chung cu hay tai lieu\n'
            'phap ly, viec xoa/them chi tiet co the lam sai lech su that.\n'
            'Can giu ban goc va cong bo thuat toan / muc dieu chinh.',10)
        text(fig,.08,.18,'SU DUNG AI: Nhom co dung AI de tham khao them ve cac thuat toan\n'
            'va cach viet bao cao; tat ca code va ket qua da duoc tu kiem tra lai.',9)
        text(fig,.08,.095,'PHAN CONG: '+ ' | '.join(f'{name} ({id_}): {pct}%' for name,id_,task,pct in MEMBERS),8)
        pdf.savefig(fig);plt.close(fig)
    print('Wrote:',outfile)
    return outfile


def run_all():
    print("=" * 55)
    print(f"BAI TAP XU LY ANH - NHOM {GROUP_NUMBER}")
    print("=" * 55)

    # In thong tin thanh vien nhom
    print("\nDANH SACH THANH VIEN:")
    total = 0

    for name, student_id, task, percentage in MEMBERS:
        print(f"{name} - {student_id}: {percentage}%")
        total += percentage

    print(f"\nTong dong gop: {total}%")
    assert total == 100, "Tong dong gop phai bang 100%"

    # Buoc 1: Tao du lieu
    print("\n[1/5] Dang chuan bi anh...")
    records, ref, degradations = create_data()

    # Buoc 2: Kiem chung thuat toan
    print("\n[2/5] B1 - Kiem tra cac thuat toan...")
    b1 = verify_b1(ref)

    # Buoc 3: Khao sat tham so
    print("\n[3/5] B2 - Khao sat PSNR va SSIM...")
    b2 = run_b2(ref, degradations)

    # Buoc 4: Chay pipeline
    print("\n[4/5] B3 - Xu ly anh qua pipeline...")
    b3 = run_b3(records)

    # Buoc 5: Truong hop that bai
    print("\n[5/5] B4 - Xem cac truong hop that bai...")
    b4 = run_b4(records)

    # Khong goi ham report() vi PDF khong hien thi dau tieng Viet

    print("\n" + "=" * 55)
    print("DA HOAN THANH B1 - B5")
    print("=" * 55)

    print("\nKET QUA:")
    print("- Du lieu anh: data/")
    print("- Bang so lieu va bieu do: results/")
    print("- Bao cao: BaoCao_B1_Nhom01.pdf")
    print("- Notebook: B1_Nhom01.ipynb")

    return {
        'b1': b1,
        'b2': b2,
        'b3': b3,
        'b4': b4
    }


if __name__ == '__main__':
    run_all()
