"""
AlgoRythms - Agent Double IA Multimodal Musical
Developpe par Dream Stream Sciences
"""

import base64
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from agent import (
    run_agent, load_models, load_features_data,
    get_recommendations, explain_with_claude,
    translate_feature, SPECTRO_DIR, CNN_DIR
)

# =============================================
# CONFIG & STYLE
# =============================================
st.set_page_config(page_title='AlgoRythms', page_icon='D', layout='wide',
                   initial_sidebar_state='collapsed')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1200px; }
section[data-testid="stSidebar"],
button[data-testid="stSidebarCollapseButton"] { display: none; }
div[data-testid="stRadioGroup"] { justify-content: center; }
div[data-testid="stTabs"] button { flex: 1; }
div[data-testid="stMarkdownContainer"] { text-align: center; }
div[data-testid="stCaptionContainer"] { text-align: center; }
h1, h2, h3, h4 { text-align: center; }
button[data-testid="stBaseButton-secondary"] {
    font-size: 0.75rem !important; padding: 8px 14px !important;
    border-radius: 6px !important; border: 1px solid #333 !important;
    background-color: #161616 !important; color: #bbb !important;
}
button[data-testid="stBaseButton-secondary"]:hover {
    background-color: #222 !important; color: #fff !important;
    border-color: #555 !important;
}
.section-title {
    font-size: 0.9rem; color: #aaa; font-weight: 500; text-align: center;
    padding: 8px 16px; margin: 16px 0 12px 0;
    border: 1px solid #222; border-radius: 6px; background-color: #111;
}
.footer {
    text-align: center; color: #888; font-size: 0.85rem;
    margin-top: 3rem; padding: 1.5rem 0; border-top: 1px solid #222;
}
</style>
""", unsafe_allow_html=True)

# =============================================
# CHARGEMENT
# =============================================
try:
    m = load_models()
    df_feat = load_features_data()
    genres = m['genres']
except Exception as e:
    st.error(f'Erreur : {e}')
    st.caption("Lancez NB9 d'abord.")
    st.stop()

# =============================================
# HEADER
# =============================================
IMG_DIR = Path(__file__).parent / 'images'
logo_path = IMG_DIR / 'vecteezy_sound-wave-light-particles-sound-spectrum-dance_33164292.png'
with open(logo_path, 'rb') as f:
    logo_b64 = base64.b64encode(f.read()).decode()

casque_path = IMG_DIR / 'favpng_c0c3bf5905f226c297cb84553cd58360.png'
with open(casque_path, 'rb') as f:
    casque_b64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
<div style="text-align:center; margin-bottom:1.5rem">
    <div style="font-size:2.5rem; font-weight:700; color:#fff; letter-spacing:-0.03em">AlgoRythms</div>
    <div style="font-size:1.3rem; color:#999; font-weight:400; margin-top:4px">Agent Double IA Multimodal Musical</div>
    <img src="data:image/png;base64,{logo_b64}" style="width:500px; margin:16px auto; display:block; opacity:0.85">
    <div style="color:#999; font-size:0.85rem; letter-spacing:0.12em; text-transform:uppercase; font-weight:500">
        Developpe par Dream Stream Sciences</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# ONGLETS
# =============================================
tab_analyse, tab_reco, tab_explain, tab_about = st.tabs([
    'Analyse', 'Recommandation', 'Explicabilite', 'Projet'
])

# =============================================
# ANALYSE
# =============================================
with tab_analyse:
    _, col_radio, _ = st.columns([1, 2, 1])
    with col_radio:
        input_mode = st.radio('', ['Piste FMA', 'Fichier audio'],
                              horizontal=True, label_visibility='collapsed')

    track_id = None
    mp3_path = None

    if input_mode == 'Piste FMA':
        cols = st.columns([1] * (len(genres) + 1))
        with cols[0]:
            if st.button('Tous', use_container_width=True):
                st.session_state['genre_filter'] = 'Tous'
        short = {'Experimental': 'Experim.', 'International': 'Internat.',
                 'Instrumental': 'Instrum.', 'Electronic': 'Electro.'}
        for i, g in enumerate(genres):
            with cols[i + 1]:
                if st.button(short.get(g, g), use_container_width=True):
                    st.session_state['genre_filter'] = g

        genre_filter = st.session_state.get('genre_filter', 'Tous')
        df_f = df_feat[df_feat['genre_top'] == genre_filter] if genre_filter != 'Tous' else df_feat

        _, col_c, _ = st.columns([1, 3, 1])
        with col_c:
            options = [(f'{r.get("track_title","")}  --  {r.get("artist_name","")}  [{r["genre_top"]}]',
                        int(r['track_id'])) for _, r in df_f.head(500).iterrows()]
            if options:
                sel = st.selectbox('Piste', [o[0] for o in options],
                                   index=None, placeholder='Choisissez votre piste',
                                   label_visibility='collapsed')
                if sel:
                    track_id = dict(options)[sel]
    else:
        _, col_c, _ = st.columns([1, 3, 1])
        with col_c:
            uploaded = st.file_uploader('', type=['mp3', 'wav', 'ogg'],
                                        label_visibility='collapsed')
            if uploaded:
                audio_bytes = uploaded.read()
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                    tmp.write(audio_bytes)
                    mp3_path = tmp.name
                st.session_state['uploaded_name'] = uploaded.name
                st.session_state['uploaded_bytes'] = audio_bytes

    _, col_b, _ = st.columns([1, 3, 1])
    with col_b:
        clicked = st.button('Analyser', use_container_width=True)

    can_go = track_id is not None or mp3_path is not None
    if clicked and can_go:
        with st.spinner(''):
            r1 = run_agent(track_id=track_id, mp3_path=mp3_path, mode='V1')
            r2 = run_agent(track_id=track_id, mp3_path=mp3_path, mode='V2')
        if 'uploaded_name' in st.session_state and mp3_path is not None:
            real_name = Path(st.session_state['uploaded_name']).stem
            r1['title'] = real_name
            r2['title'] = real_name
        st.session_state['r1'] = r1
        st.session_state['r2'] = r2

    if 'r1' in st.session_state:
        r1 = st.session_state['r1']
        r2 = st.session_state['r2']

        # Lecteur audio
        audio_bytes_play = None
        if r1.get('track_id'):
            tid = int(r1['track_id'])
            folder = f'{tid:06d}'[:3]
            ap = Path(__file__).parent / 'data' / 'raw' / 'fma_small' / 'fma_small' / folder / f'{tid:06d}.mp3'
            if ap.exists():
                with open(ap, 'rb') as af:
                    audio_bytes_play = af.read()
        elif 'uploaded_bytes' in st.session_state:
            audio_bytes_play = st.session_state['uploaded_bytes']

        genre_line = f'<div style="color:#555; font-size:0.8rem; margin-top:2px">Genre reel : {r1["true_genre"]}</div>' if r1["true_genre"] != "?" else ""
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:14px; margin:8px 0; text-align:left">
            <img src="data:image/png;base64,{casque_b64}" style="width:80px; height:80px; object-fit:contain; opacity:0.8">
            <div>
                <div style="font-size:1.1rem; font-weight:500">{r1["title"]}</div>
                <div style="color:#888">{r1["artist"]}</div>
                {genre_line}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if audio_bytes_play:
            st.audio(audio_bytes_play, format='audio/mp3')

        # Resultats V1 / V2
        col1, col2 = st.columns(2)
        for col, r, label, accent in [(col1, r1, 'Classification Agent N\u00b01', '#1DB954'),
                                       (col2, r2, 'Classification Agent N\u00b02', '#4a9eff')]:
            with col:
                ok = r['pred_genre'] == r['true_genre']
                conf = r['confidence'] * 100
                verdict = 'correct' if ok else 'erreur' if r['true_genre'] != '?' else ''
                bar_color = '#1DB954' if ok else accent

                st.markdown(f'<div class="section-title">{label}</div>', unsafe_allow_html=True)
                st.subheader(r['pred_genre'])
                st.markdown(f"""<div style="margin:8px 0 4px 0">
<div style="display:flex; justify-content:space-between; margin-bottom:3px">
<span style="color:#888; font-size:0.75rem">confiance</span>
<span style="color:#fff; font-size:0.8rem; font-weight:500">{conf:.0f}%</span>
</div>
<div style="background:#1a1a1a; border-radius:3px; height:6px">
<div style="background:{bar_color}; width:{min(conf,100)}%; height:6px; border-radius:3px"></div>
</div>
<div style="color:#666; font-size:0.7rem; margin-top:3px">{verdict}</div>
</div>""", unsafe_allow_html=True)

                st.markdown('<div style="color:#666; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:16px">Top 3 genres probables</div>', unsafe_allow_html=True)
                for rank, (g, p) in enumerate(r['top_k'][:3], 1):
                    pct = min(p * 100, 100)
                    opacity = '1' if rank == 1 else '0.6'
                    st.markdown(f"""<div style="display:flex; align-items:center; margin:6px 0; opacity:{opacity}">
<span style="color:#aaa; font-size:0.8rem; width:120px">{g}</span>
<div style="flex:1; background:#1a1a1a; border-radius:3px; height:5px; margin:0 10px">
<div style="background:{accent}; width:{pct}%; height:5px; border-radius:3px"></div>
</div>
<span style="color:#888; font-size:0.75rem; width:30px; text-align:right">{pct:.0f}%</span>
</div>""", unsafe_allow_html=True)

        # Flags
        flags = []
        if r2.get('mismatch'):
            flags.append('Mismatch sous-genres')
        if r2.get('suspect'):
            flags.append('Piste acoustiquement atypique')
        if 'Blue Dot' in str(r2.get('artist', '')):
            flags.append('Blue Dot Sessions (biais corpus)')
        if flags:
            st.caption(' | '.join(flags))

# =============================================
# RECOMMANDATION
# =============================================
with tab_reco:
    if 'r1' not in st.session_state:
        st.caption("Analysez d'abord une piste.")
    else:
        r = st.session_state['r1']
        reco = r.get('reco', [])
        if not reco:
            r = st.session_state.get('r2', {})
            reco = r.get('reco', [])

        if reco:
            _, col_banner, _ = st.columns([1, 2, 1])
            with col_banner:
                st.image(str(IMG_DIR / 'group-people-enjoying-holi-color.jpg'),
                         use_container_width=True)

            # Lecteur audio
            r_cur = st.session_state['r1']
            if r_cur.get('track_id'):
                tid_c = int(r_cur['track_id'])
                folder_c = f'{tid_c:06d}'[:3]
                ap_c = Path(__file__).parent / 'data' / 'raw' / 'fma_small' / 'fma_small' / folder_c / f'{tid_c:06d}.mp3'
                if ap_c.exists():
                    _, col_pl, _ = st.columns([1, 2, 1])
                    with col_pl:
                        with open(ap_c, 'rb') as af_c:
                            st.audio(af_c.read(), format='audio/mp3')

            st.markdown(f'<div class="section-title">Similaires a : {r["title"]} -- {r["artist"]}</div>',
                        unsafe_allow_html=True)

            for rank, rec in enumerate(reco, 1):
                sim = rec['similarity']
                bar_color = '#1DB954' if sim > 90 else '#4a9eff'
                col_info, col_bar, col_btn, _ = st.columns([3, 2, 1, 2])
                with col_info:
                    st.markdown(f'<div style="text-align:right"><div style="font-weight:500">{rec["title"]}</div>'
                                f'<div style="color:#666; font-size:0.8rem">{rec["artist"]} -- {rec["genre"]}</div></div>',
                                unsafe_allow_html=True)
                with col_bar:
                    st.markdown(f'<div style="display:flex; align-items:center; gap:8px; padding-top:8px">'
                                f'<div style="flex:1; background:#1a1a1a; border-radius:3px; height:5px">'
                                f'<div style="background:{bar_color}; width:{min(sim,100)}%; height:5px; border-radius:3px"></div>'
                                f'</div><span style="color:#ccc; font-size:0.8rem">{sim:.0f}%</span></div>',
                                unsafe_allow_html=True)
                with col_btn:
                    if st.button('Tester', key=f'r_{rank}'):
                        with st.spinner(''):
                            r1_new = run_agent(track_id=rec['track_id'], mode='V1')
                            r2_new = run_agent(track_id=rec['track_id'], mode='V2')
                        st.session_state['r1'] = r1_new
                        st.session_state['r2'] = r2_new
                        st.rerun()
        else:
            st.caption('Non disponible pour cette piste.')

# =============================================
# EXPLICABILITE
# =============================================
with tab_explain:
    if 'r2' not in st.session_state:
        st.caption("Analysez d'abord une piste.")
    else:
        r2 = st.session_state['r2']
        shap_features = r2.get('shap_features', [])

        if shap_features:
            st.markdown('<div class="section-title">Ce que le modele a entendu</div>',
                        unsafe_allow_html=True)
            max_shap = max(abs(sv) for _, sv in shap_features[:6])

            for feat_name, shap_val in shap_features[:6]:
                feat_fr = translate_feature(feat_name)
                direction = r2['pred_genre'] if shap_val > 0 else 'contre'
                bar_width = min(abs(shap_val) / max_shap * 100, 100)
                bar_color = '#4a9eff' if shap_val > 0 else '#444'
                pct = abs(shap_val) / max_shap * 100

                st.markdown(f"""<div style="display:flex; align-items:center; margin:5px 0">
<div style="width:50%; text-align:right; padding-right:12px">
<span style="font-size:0.8rem; color:#aaa">{feat_fr}</span></div>
<div style="width:50%; display:flex; align-items:center; gap:8px">
<div style="flex:1; background:#1a1a1a; border-radius:3px; height:5px">
<div style="background:{bar_color}; width:{bar_width}%; height:5px; border-radius:3px"></div></div>
<span style="color:#888; font-size:0.75rem; width:50px; text-align:right">{pct:.0f}%</span>
<span style="color:#555; font-size:0.65rem; width:50px">{direction}</span>
</div></div>""", unsafe_allow_html=True)

        # Grad-CAM
        tid = r2.get('track_id')
        if tid:
            spec_path = SPECTRO_DIR / f'{int(tid):06d}.npy'
            model_path = CNN_DIR / 'best_audio_cnn.pth'

            if spec_path.exists() and model_path.exists():
                st.markdown('<div class="section-title">Grad-CAM -- O\u00f9 le CNN regarde dans le spectrogramme</div>',
                            unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.75rem; color:#aaa; text-align:center; margin-bottom:8px">'
                            'Les zones chaudes (jaune/rouge) sont celles qui activent le plus la decision du CNN. '
                            'Axe horizontal : temps (0-30s). Axe vertical : frequences (graves en bas, aigus en haut).'
                            '</div>', unsafe_allow_html=True)
                try:
                    import torch, torch.nn as nn, torch.nn.functional as F
                    import matplotlib.pyplot as plt, matplotlib.cm as cm_mpl, cv2

                    class AudioCNN(nn.Module):
                        def __init__(self, nc):
                            super().__init__()
                            self.conv1 = nn.Conv2d(1,16,3,padding=1); self.bn1 = nn.BatchNorm2d(16)
                            self.conv2 = nn.Conv2d(16,32,3,padding=1); self.bn2 = nn.BatchNorm2d(32)
                            self.conv3 = nn.Conv2d(32,64,3,padding=1); self.bn3 = nn.BatchNorm2d(64)
                            self.conv4 = nn.Conv2d(64,128,3,padding=1); self.bn4 = nn.BatchNorm2d(128)
                            self.pool = nn.MaxPool2d(2,2); self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4))
                            self.fc1 = nn.Linear(128*4*4,512); self.fc2 = nn.Linear(512,256)
                            self.dropout = nn.Dropout(0.3); self.fc3 = nn.Linear(256,nc)
                        def forward(self, x):
                            x = self.pool(F.relu(self.bn1(self.conv1(x))))
                            x = self.pool(F.relu(self.bn2(self.conv2(x))))
                            x = self.pool(F.relu(self.bn3(self.conv3(x))))
                            x = self.adaptive_pool(F.relu(self.bn4(self.conv4(x))))
                            x = x.view(x.size(0),-1)
                            x = F.relu(self.fc1(x)); x = self.dropout(x)
                            x = F.relu(self.fc2(x)); return self.fc3(x)

                    class GradCAM:
                        def __init__(self, model, layer):
                            self.model = model; self.g = None; self.a = None
                            layer.register_forward_hook(lambda m,i,o: setattr(self,'a',o.detach()))
                            layer.register_full_backward_hook(lambda m,gi,go: setattr(self,'g',go[0].detach()))
                        def generate(self, t, tc=None):
                            self.model.zero_grad(); out = self.model(t)
                            if tc is None: tc = out.argmax(1).item()
                            oh = torch.zeros_like(out); oh[0,tc] = 1.0; out.backward(gradient=oh)
                            w = self.g.mean(dim=(2,3),keepdim=True)
                            c = F.relu((w*self.a).sum(dim=1).squeeze())
                            mn,mx = c.min(),c.max()
                            if mx-mn > 1e-8: c = (c-mn)/(mx-mn)
                            return c.cpu().numpy(), tc

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    cnn = AudioCNN(len(genres)).to(device)
                    cnn.load_state_dict(torch.load(model_path, map_location=device))
                    cnn.eval()
                    gc = GradCAM(cnn, cnn.conv4)

                    spec = np.load(spec_path)
                    TF = 1291
                    spec = spec[:,:TF] if spec.shape[1] >= TF else np.pad(spec,((0,0),(0,TF-spec.shape[1])))
                    sn = (spec - spec.mean()) / (spec.std() + 1e-8)
                    tensor = torch.tensor(sn, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    cam, pc = gc.generate(tensor)
                    cam_r = cv2.resize(cam, (TF,128), interpolation=cv2.INTER_LINEAR)
                    hm = cm_mpl.jet(cam_r)[:,:,:3]
                    s_min, s_max = spec.min(), spec.max()
                    sr = cm_mpl.magma((spec-s_min)/(s_max-s_min+1e-8))[:,:,:3]
                    overlay = np.clip(0.6*sr + 0.4*hm, 0, 1)
                    pred_label = m['le'].classes_[pc]

                    # Superposition
                    fig1, ax1 = plt.subplots(figsize=(10,2.5))
                    ax1.imshow(overlay, aspect='auto', origin='lower', extent=[0,30,0,128])
                    ax1.set_title(f'Grad-CAM — {pred_label}', fontsize=9, color='#eee')
                    ax1.set_xlabel('temps (s)', fontsize=7, color='#bbb')
                    ax1.set_ylabel('freq', fontsize=7, color='#bbb')
                    ax1.tick_params(labelsize=6, colors='#aaa')
                    ax1.set_facecolor('#0e1117'); fig1.set_facecolor('#0e1117')
                    for s in ax1.spines.values(): s.set_visible(False)
                    plt.tight_layout(); st.pyplot(fig1); plt.close()

                    # Texte decomposition
                    st.markdown('<div style="font-size:0.75rem; color:#aaa; text-align:center; margin:8px 0">'
                                'Decomposition : a gauche, le spectrogramme brut (empreinte sonore). '
                                'A droite, la carte Grad-CAM (zones decisives pour le CNN).</div>',
                                unsafe_allow_html=True)

                    # Decomposition
                    fig3, axes = plt.subplots(1, 2, figsize=(10,2))
                    axes[0].imshow(spec, aspect='auto', origin='lower', cmap='magma', extent=[0,30,0,128])
                    axes[0].set_title('Spectrogramme', fontsize=7, color='#ccc')
                    axes[1].imshow(cam_r, aspect='auto', origin='lower', cmap='jet', vmin=0, vmax=1, extent=[0,30,0,128])
                    axes[1].set_title('Grad-CAM', fontsize=7, color='#ccc')
                    for ax in axes:
                        ax.tick_params(labelsize=4, colors='#aaa')
                        ax.set_facecolor('#0e1117')
                        for s in ax.spines.values(): s.set_visible(False)
                    fig3.set_facecolor('#0e1117')
                    plt.tight_layout(); st.pyplot(fig3); plt.close()
                except Exception as e:
                    st.caption(f'Grad-CAM non disponible : {e}')

        # Claude
        _, col_claude, _ = st.columns([1, 2, 1])
        with col_claude:
            if st.button('Generer explication Claude (optionnel)', use_container_width=True):
                with st.spinner(''):
                    expl = explain_with_claude(r2, mode='V2')
                    expl_clean = expl.replace('**','').replace('*','').replace('#','').replace('`','')
                st.markdown(f'<div style="font-size:0.8rem; color:#bbb; line-height:1.6">{expl_clean}</div>',
                            unsafe_allow_html=True)

# =============================================
# PROJET
# =============================================
with tab_about:
    _, col_img_proj, _ = st.columns([2, 2, 2])
    with col_img_proj:
        st.image(str(IMG_DIR / '\u2014Pngtree\u2014woman wearing headphones enjoying music_19429199.jpg'),
                 use_container_width=True)

    st.markdown('<div class="section-title">Classement global</div>', unsafe_allow_html=True)
    results_data = []
    for csv_file in sorted((Path(__file__).parent / 'outputs' / 'resultats').glob('results_nb*.csv')):
        df_r = pd.read_csv(csv_file)
        for _, row in df_r.iterrows():
            results_data.append({
                'Source': csv_file.stem.replace('results_', '').upper(),
                'Modele': row.get('model', '?'),
                'F1': round(row.get('f1_test', 0), 4),
                'Acc': round(row.get('acc_test', 0), 4),
            })
    if results_data:
        df_results = pd.DataFrame(results_data).sort_values('F1', ascending=False)
        st.dataframe(df_results, use_container_width=True, hide_index=True,
                     height=(len(df_results)+1)*35+3)

    st.markdown('<div class="section-title">Architecture</div>', unsafe_allow_html=True)
    df_arch = pd.DataFrame([
        {'Phase': 'EDA', 'Notebooks': 'NB1', 'Approche': 'Exploration, mismatch, biais'},
        {'Phase': 'Features', 'Notebooks': 'NB2, NB2BIS', 'Approche': '351 features maison'},
        {'Phase': 'ML tabulaire', 'Notebooks': 'NB3, NB3BIS, NB3TER', 'Approche': 'LR, RF, XGBoost, MLP'},
        {'Phase': 'Deep Learning', 'Notebooks': 'NB4', 'Approche': 'CNN log-mel'},
        {'Phase': 'Mismatch', 'Notebooks': 'NB5', 'Approche': 'Multi-label, sous-genres'},
        {'Phase': 'Comparaison V1', 'Notebooks': 'NB6', 'Approche': '6 modeles + recommandation'},
        {'Phase': 'Transfer Learning', 'Notebooks': 'NB7, NB7 (8000)', 'Approche': 'PANNs CNN14'},
        {'Phase': 'NLP', 'Notebooks': 'NB8', 'Approche': 'TF-IDF metadonnees'},
        {'Phase': 'Comparaison V2', 'Notebooks': 'NB6BIS', 'Approche': 'Tous modeles + curation'},
        {'Phase': 'Interpretabilite', 'Notebooks': 'NB6TER', 'Approche': 'SHAP, Grad-CAM, biais'},
        {'Phase': 'Agent IA', 'Notebooks': 'NB9', 'Approche': 'Ensemble V1/V2 + Streamlit'},
    ])
    st.dataframe(df_arch, use_container_width=True, hide_index=True,
                 height=(len(df_arch)+1)*35+3)

# Footer
st.markdown(
    '<div class="footer">AlgoRythms -- Projet ML2 Sorbonne Data Analytics 2026 | '
    'FMA Small (Defferrard et al., ISMIR 2017)</div>',
    unsafe_allow_html=True
)
