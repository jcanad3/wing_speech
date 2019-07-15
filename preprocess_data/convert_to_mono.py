import os, glob

for species_dir in glob.glob('../train_2_birds_mp3/*'):
	species_path = os.path.basename(species_dir)
	os.makedirs('../train_2_birds_wavs/' + species_path, exist_ok=True)
	for recording in glob.glob(species_dir + '/*'):
		name = os.path.basename(recording)
		name = name.replace('mp3', 'wav')
		if name not in os.listdir('../train_2_birds_wavs/' + species_path):
			print(name)
			os.system('ffmpeg -i ' + recording + ' -ar 16000 -ac 1 ../train_2_birds_wavs/' + species_path + '/' + name)
