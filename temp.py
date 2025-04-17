from speechbrain.inference.VAD import VAD

VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="models/vad-crdnn-libriparty")
# boundaries = VAD.get_speech_segments(audio_file)
# VAD.save_boundaries(boundaries)