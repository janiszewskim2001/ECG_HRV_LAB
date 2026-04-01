
import pandas as pd
import numpy as np 
import glob, os
from scipy.signal import savgol_filter, find_peaks
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import emd

#%%--------------------------------Ustawienia i Ścieżki------------------------
path_inp = r"/Users/michaljaniszewski/Desktop/FIZYKA MEDYCZNA/sem 2 /zaaw. lab fizyki med/EKG LAB"
if os.path.exists(path_inp):
    os.chdir(path_inp)
else:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
st.set_page_config(layout="wide", page_title="Analiza EKG i EMD")

# Kolory (Nasz Dark Mode)
kolor_sygnalu = "#00FFAA"  # Neonowa Morska Zieleń
kolor_analizy = "#FF9900"  # Pomarańcz
ciemny_szary  = "#2d3436"
tlo_glowne    = "#0e1117"
lekki_szary   = "#1a1c23"
bialo_szary   = "#a0a0a0"
bialy         = "#ffffff"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {tlo_glowne}; color: {bialo_szary}; }}
    h1, h2, h3, [data-testid="stHeader"], .section-title span {{ color: {bialy} !important; }}
    [data-testid="stMetricValue"] {{ color: {kolor_analizy} !important; font-size: 32px !important; font-weight: bold; }}
    [data-testid="stMetricLabel"] p {{ color: {bialy} !important; }}
    .moja-ramka {{ border-radius: 10px; padding: 25px; background-color: {lekki_szary}; border-left: 5px solid {kolor_sygnalu}; border-right: 5px solid {kolor_analizy}; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }}
    .moja-ramka h4 {{ color: {bialy}; margin: 0; font-size: 28px; font-weight: bold;}}
    .section-title {{ font-size: 22px; font-weight: bold; margin-top: 20px; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 2px solid {ciemny_szary}; }}
    </style>
    """, unsafe_allow_html=True)
    
# Funkcja do nakładania naszego mrocznego stylu na wykresy (odchudza kod!)
def aplikuj_ciemny_motyw(figura, wys=350):
    figura.update_layout(
        height=wys, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=bialo_szary),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return figura

#%%--------------------------------Ładowanie danych----------------------------
@st.cache_data
def load_my_data(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    data_list = []
    for line in lines:
        parts = line.replace(',', '.').split()
        if len(parts) >= 2:
            try: data_list.append([float(parts[0]), float(parts[1])])
            except ValueError: continue
    return pd.DataFrame(data_list, columns=['czas', 'ecg'])

txt_files = glob.glob("*.txt")
if not txt_files:
    st.warning("Brak plików .txt w folderze.")
    st.stop()

df_all = load_my_data(st.sidebar.selectbox("Wybierz plik z danymi EKG:", txt_files))

st.markdown('<div class="moja-ramka"><h4>Zaawansowana Analiza HRV i EMD sygnału EKG</h4><p>Laboratorium fizyki medycznej</p></div>', unsafe_allow_html=True)   

#%%--------------------------------SEKCJA 1: PODGLĄD---------------------------
st.markdown('<div class="section-title"><span>1) Podgląd danych surowych</span></div>', unsafe_allow_html=True)

min_czas, max_czas = float(df_all['czas'].min()), float(df_all['czas'].max())
zakres_czasu = st.slider("Zakres czasu do analizy [s]:", min_czas, max_czas, (min_czas, min(min_czas + 20.0, max_czas)), 0.1)

df = df_all[(df_all['czas'] >= zakres_czasu[0]) & (df_all['czas'] <= zakres_czasu[1])].copy()

if len(df) > 28000:
    st.warning("Uwaga: W sekcji EMD zostanie przetworzone tylko pierwsze 28000 próbek ze względu na obciążenie procesora.")

colA, colB = st.columns(2)
with colA:
    with st.container(border=True): st.dataframe(df, height=200, use_container_width=True)
with colB:
    with st.container(border=True):
        fig_pie = px.pie(names=["Fragment do analizy", "Pozostała część"], values=[len(df), len(df_all) - len(df)], hole=0.5, color_discrete_sequence=[kolor_sygnalu, "#444444"])
        fig_pie.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', font=dict(color=bialo_szary))
        st.plotly_chart(fig_pie, use_container_width=True)

with st.container(border=True):
    fig_glowny = go.Figure()
    fig_glowny.add_trace(go.Scatter(x=df_all['czas'], y=df_all['ecg'], name='Pozostała część', line=dict(color="#444444", width=1)))
    fig_glowny.add_trace(go.Scatter(x=df['czas']-0.3*np.sin(5*df['czas'],y=df_all['ecg'])))
    fig_glowny.add_trace(go.Scatter(x=df['czas'], y=df['ecg'], name='Fragment do analizy', line=dict(color=kolor_sygnalu, width=2)))
    fig_glowny = aplikuj_ciemny_motyw(fig_glowny)
    fig_glowny.update_layout(xaxis_title="Czas [s]", yaxis_title="Amplituda [mV]")
    st.plotly_chart(fig_glowny, use_container_width=True)

#%%--------------------------------SEKCJA 2: FILTROWANIE-----------------------
st.markdown('<div class="section-title"><span>2) Filtracja sygnału</span></div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
win_len = c1.slider("Długość okna filtra", 11, 201, 51, 2)
poly = c2.slider("Stopień wielomianu", 1, 5, 3, 1)

win_len = win_len + 1 if win_len % 2 == 0 else win_len
win_len = max(win_len, poly + 2)
win_len = win_len + 1 if win_len % 2 == 0 else win_len

df['ecg_filtered'] = savgol_filter(df['ecg'], window_length=win_len, polyorder=poly)

with st.container(border=True):
    fig_filt = go.Figure()
    fig_filt.add_trace(go.Scatter(x=df['czas'], y=df['ecg'], name='Surowy', line=dict(color="#444444", width=1.5)))
    fig_filt.add_trace(go.Scatter(x=df['czas'], y=df['ecg_filtered'], name='Przefiltrowany', line=dict(color=kolor_sygnalu, width=2.5)))
    st.plotly_chart(aplikuj_ciemny_motyw(fig_filt).update_layout(xaxis_title="Czas [s]"), use_container_width=True)

#%%--------------------------------SEKCJA 3: HRV-------------------------------
st.markdown('<div class="section-title"><span>3) Analiza HRV</span></div>', unsafe_allow_html=True)

peaks, _ = find_peaks(df['ecg_filtered'], distance=400, prominence=0.1)

if len(peaks) > 1:
    czas_pikow = df['czas'].iloc[peaks].values
    rr_intervals = np.diff(czas_pikow) * 1000  
    
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Średnie Tętno (BPM)", f"{60000 / np.mean(rr_intervals):.0f} ud./min")
    with m2: st.metric("SDNN (Zmienność)", f"{np.std(rr_intervals):.1f} ms")
    with m3: st.metric("RMSSD (Przywspółczulna)", f"{np.sqrt(np.mean(np.diff(rr_intervals)**2)):.1f} ms")

    with st.container(border=True):
        fig_peaks = go.Figure()
        fig_peaks.add_trace(go.Scatter(x=df['czas'], y=df['ecg_filtered'], line=dict(color=kolor_sygnalu, width=1.5), name='EKG'))
        fig_peaks.add_trace(go.Scatter(x=czas_pikow, y=df['ecg_filtered'].iloc[peaks], mode='markers', marker=dict(color=kolor_analizy, size=8), name='Załamki R'))
        st.plotly_chart(aplikuj_ciemny_motyw(fig_peaks, 300), use_container_width=True)

    c_tach, c_hist = st.columns(2)
    with c_tach:
        with st.container(border=True):
            st.markdown("<p style='text-align:center;'><b>Tachogram</b></p>", unsafe_allow_html=True)
            fig_rr = go.Figure(go.Scatter(x=czas_pikow[1:], y=rr_intervals, mode='lines+markers', line=dict(color=kolor_analizy)))
            st.plotly_chart(aplikuj_ciemny_motyw(fig_rr, 250), use_container_width=True)
    with c_hist:
        with st.container(border=True):
            st.markdown("<p style='text-align:center;'><b>Histogram RR</b></p>", unsafe_allow_html=True)
            fig_hist = px.histogram(x=rr_intervals, nbins=20, color_discrete_sequence=[kolor_analizy])
            fig_hist.update_traces(marker_line_color=bialy, marker_line_width=0.5)
            st.plotly_chart(aplikuj_ciemny_motyw(fig_hist, 250).update_layout(bargap=0.1), use_container_width=True)
else:
    st.warning("Zbyt mało danych do analizy HRV.")

#%%--------------------------------SEKCJA 4: QRS-------------------------------
st.markdown('<div class="section-title"><span>4) Segmentacja zespołu QRS</span></div>', unsafe_allow_html=True)

fs, pre_w, post_w = 1000, 200, 400
qrs_epochs = [df['ecg_filtered'].iloc[p - pre_w : p + post_w].values for p in peaks if p - pre_w >= 0 and p + post_w < len(df)]

if qrs_epochs:
    time_qrs = np.linspace(-0.2, 0.4, pre_w + post_w)
    fig_qrs = go.Figure()
    for ep in qrs_epochs:
        fig_qrs.add_trace(go.Scatter(x=time_qrs, y=ep, mode='lines', line=dict(color=kolor_sygnalu, width=1), opacity=0.1, showlegend=False))
    fig_qrs.add_trace(go.Scatter(x=time_qrs, y=np.mean(qrs_epochs, axis=0), mode='lines', line=dict(color=kolor_analizy, width=4), name='Średni profil'))
    with st.container(border=True):
        st.plotly_chart(aplikuj_ciemny_motyw(fig_qrs, 400).update_layout(xaxis_title="Czas od R [s]"), use_container_width=True)

#%%--------------------------------SEKCJA 5: EMD-------------------------------
st.markdown('<div class="section-title"><span>5) Dekompozycja EMD, usunięcie oddechu i widmo FFT</span></div>', unsafe_allow_html=True)

with st.spinner("Algorytm EMD przelicza dane (Może to chwilę potrwać przy zmianie suwaka)..."):
    
    limit = min(len(df), 28000)
    ecg_arr, czas_arr = df['ecg_filtered'].values[:limit], df['czas'].values[:limit]
    
    # OBLICZENIA EMD
    imf = emd.sift.sift(ecg_arr)
    liczba_warstw = imf.shape[1]
    
    # 1. DRABINKA IMF
    with st.container(border=True):
        st.markdown("<p style='text-align:center;'><b>Warstwy IMF od najwyższej do najniższej częstotliwości</b></p>", unsafe_allow_html=True)
        fig_imf = make_subplots(rows=liczba_warstw, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        kolory = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
        
        for i in range(liczba_warstw):
            nazwa = f"IMF {i+1}" + (" (Oddech)" if i >= liczba_warstw - 2 else "")
            fig_imf.add_trace(go.Scatter(x=czas_arr, y=imf[:, i], name=nazwa, line=dict(color=kolory[i % len(kolory)], width=1.5)), row=i+1, col=1)
            fig_imf.update_yaxes(title_text=f"IMF {i+1}", showticklabels=False, row=i+1, col=1)

        st.plotly_chart(aplikuj_ciemny_motyw(fig_imf, max(500, 120 * liczba_warstw)).update_layout(xaxis_title="Czas [s]"), use_container_width=True)
    
    # 2. KOREKTA LINII (WYPROSTOWANY SYGNAŁ)
    oddech = imf[:, -1] + imf[:, -2]
    ecg_wyprostowane = ecg_arr - oddech
    
    with st.container(border=True):
        st.markdown("<p style='text-align:center;'><b>Sygnał po korekcie linii izoelektrycznej</b></p>", unsafe_allow_html=True)
        fig_emd = go.Figure()
        fig_emd.add_trace(go.Scatter(x=czas_arr, y=ecg_arr, name='Przed korektą (Pływający)', line=dict(color="#444444", width=1.5)))
        fig_emd.add_trace(go.Scatter(x=czas_arr, y=ecg_wyprostowane, name='Sygnał wyprostowany (EMD)', line=dict(color=kolor_sygnalu, width=2.5)))
        st.plotly_chart(aplikuj_ciemny_motyw(fig_emd, 350).update_layout(xaxis_title="Czas [s]", yaxis_title="Amplituda [mV]"), use_container_width=True)

    # 3. ANALIZA WIDMOWA (FFT)
    with st.container(border=True):
        st.markdown("<p style='text-align:center;'><b>Analiza Widmowa (FFT) - Sprawdzenie poprawności filtracji</b></p>", unsafe_allow_html=True)
        n_fft = len(ecg_arr)
        freqs = np.fft.fftfreq(n_fft, d=1/1000)
        
        fft_raw = np.abs(np.fft.fft(ecg_arr) / n_fft)
        fft_clean = np.abs(np.fft.fft(ecg_wyprostowane) / n_fft)
        
        pos_mask = freqs > 0 
        
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(x=freqs[pos_mask], y=fft_raw[pos_mask], name="Widmo przed korektą EMD", line=dict(color="#444444", width=2)))
        fig_fft.add_trace(go.Scatter(x=freqs[pos_mask], y=fft_clean[pos_mask], name="Widmo po korekcie EMD", line=dict(color=kolor_analizy, width=2)))
        st.plotly_chart(aplikuj_ciemny_motyw(fig_fft, 350).update_layout(xaxis_range=[0, 15], xaxis_title="Częstotliwość [Hz]", yaxis_title="Amplituda"), use_container_width=True)

    # 4. EXPORT PLIKU
    st.success("Analiza zakończona, możesz zapisać wynik.")
    df_out = pd.DataFrame({'czas[s]': czas_arr, 'ECG_clean': ecg_wyprostowane})
    csv_txt = df_out.to_csv(index=False, sep='\t')
    
    st.download_button(
        label="Pobierz wyprostowane EKG (.txt)",
        data=csv_txt,
        file_name="EKG_wyprostowane_EMD.txt",
        mime="text/plain"
    )
