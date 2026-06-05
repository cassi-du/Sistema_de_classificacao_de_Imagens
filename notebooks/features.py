import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.measure import label, regionprops
from segmentacao import segmentar_grao

def extrair_descritores_cor(bgr):
    """Extrai estatísticas simples em HSV: mean e std para H,S,V."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    feats = {}
    for name, chan in (('H', h), ('S', s), ('V', v)):
        feats[f'{name}_mean'] = float(np.mean(chan))
        feats[f'{name}_std'] = float(np.std(chan))
    return feats


def extrair_descritores_textura(gray, mask):
    """Extrai LBP e GLCM (contraste, homogeneidade) sobre a região do grão.
    Retorna dicionário de features simples.
    """
    feats = {}
    region = gray.copy()
    region[mask == 0] = 0

    lbp = local_binary_pattern(region, P=8, R=1, method='uniform')
    vals = lbp[mask == 255]
    if vals.size > 0:
        for i in range(int(vals.max()) + 1):
            feats[f'lbp_{i}_ratio'] = float(np.sum(vals == i) / vals.size)
    else:
        feats['lbp_empty'] = 1.0

    img_q = (region / 32).astype(np.uint8)
    levels = 8
    h, w = img_q.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)

    for i in range(h):
        for j in range(w - 1):
            if mask[i, j] == 255 and mask[i, j + 1] == 255:
                a = int(img_q[i, j])
                b = int(img_q[i, j + 1])
                if 0 <= a < levels and 0 <= b < levels:
                    glcm[a, b] += 1.0
    if glcm.sum() == 0:
        feats['glcm_contrast'] = 0.0
        feats['glcm_homogeneity'] = 0.0
        feats['glcm_energy'] = 0.0
        feats['glcm_correlation'] = 0.0
    else:
        P = glcm / glcm.sum()
        i_idx, j_idx = np.indices(P.shape)
        contrast = np.sum(((i_idx - j_idx) ** 2) * P)
        homogeneity = np.sum(P / (1.0 + np.abs(i_idx - j_idx)))
        energy = float(np.sum(P ** 2))

        # correlation
        mu_i = np.sum(i_idx * P)
        mu_j = np.sum(j_idx * P)
        std_i = np.sqrt(np.sum(((i_idx - mu_i) ** 2) * P))
        std_j = np.sqrt(np.sum(((j_idx - mu_j) ** 2) * P))
        if std_i < 1e-10 or std_j < 1e-10:
            correlation = 0.0
        else:
            correlation = float(np.sum(((i_idx - mu_i) * (j_idx - mu_j) * P) / (std_i * std_j)))

        feats['glcm_contrast'] = float(contrast)
        feats['glcm_homogeneity'] = float(homogeneity)
        feats['glcm_energy'] = energy
        feats['glcm_correlation'] = correlation

    return feats


def features_grao(caminho):
    """Extrai 11 descritores para um grão a partir do caminho da imagem.

    Retorna dicionário com as chaves:
    H_mean, S_mean, V_mean, frac_dark,
    area, circularity, solidity, eccentricity,
    glcm_contrast, glcm_homogeneity, lbp_var
    """
    bgr, gray, mask = segmentar_grao(caminho)
    feats = {}


    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    mask_bool = mask == 255
    if np.count_nonzero(mask_bool) == 0:

        for k in ['H_mean', 'S_mean', 'V_mean', 'frac_dark',
                  'area', 'circularity', 'solidity', 'eccentricity',
                  'glcm_contrast', 'glcm_homogeneity', 'lbp_var']:
            feats[k] = float('nan')
        return feats

    feats['H_mean'] = float(np.mean(H[mask_bool]))
    feats['S_mean'] = float(np.mean(S[mask_bool]))
    feats['V_mean'] = float(np.mean(V[mask_bool]))
    feats['frac_dark'] = float(np.sum(V[mask_bool] < 80) / np.count_nonzero(mask_bool))

    # Histograma de matiz com máscara (8 bins normalizados)
    mask_uint8 = mask.astype(np.uint8)
    hist_h = cv2.calcHist([H], [0], mask_uint8, [8], [0, 180])
    hist_h = hist_h.flatten()
    hist_h = hist_h / (hist_h.sum() + 1e-10)  # normaliza
    for i, val in enumerate(hist_h):
        feats[f'hist_h_{i}'] = float(val)

    lbl = label(mask_bool.astype(np.uint8))
    props = regionprops(lbl)
    if len(props) == 0:
        area = 0
        perimeter = 0
        solidity = 0
        eccentricity = 0
        extent = 0
    else:
        p = max(props, key=lambda x: x.area)
        area = float(p.area)

        perimeter = p.perimeter if hasattr(p, 'perimeter') else 0.0
        if perimeter == 0:
            circularity = 0.0
        else:
            circularity = float(4.0 * np.pi * p.area / (perimeter ** 2))
        solidity = float(p.solidity) if hasattr(p, 'solidity') else 0.0
        eccentricity = float(p.eccentricity) if hasattr(p, 'eccentricity') else 0.0
        extent = float(p.extent) if hasattr(p, 'extent') else 0.0

    feats['area'] = area
    feats['circularity'] = circularity
    feats['solidity'] = solidity
    feats['eccentricity'] = eccentricity
    feats['perimeter'] = perimeter
    feats['extent'] = extent

    # Momentos de Hu (7 valores em escala logarítmica)
    momentos = cv2.moments(mask)
    hu = cv2.HuMoments(momentos).flatten()
    for i, val in enumerate(hu):
        feats[f'hu_{i + 1}'] = float(-np.sign(val) * np.log10(abs(val) + 1e-10))

    tex = extrair_descritores_textura(gray, mask)
    feats['glcm_contrast'] = tex.get('glcm_contrast', float('nan'))
    feats['glcm_homogeneity'] = tex.get('glcm_homogeneity', float('nan'))
    feats['glcm_energy'] = tex.get('glcm_energy', float('nan'))
    feats['glcm_correlation'] = tex.get('glcm_correlation', float('nan'))

    # LBP variance
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    vals = lbp[mask_bool]
    feats['lbp_var'] = float(np.var(vals)) if vals.size > 0 else float('nan')

    return feats
