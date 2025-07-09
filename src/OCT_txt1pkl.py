import os



thickness_d = {}
thickness_std_d = {}
surface_smoothness_d = {}
surface_smoothness_std_d = {}
epidermis_attenuation_d = {}
epidermis_attenuation_std_d = {}
edj_smoothness_d = {}
edj_smoothness_std_d = {}
dermis_attenuation_d = {}
dermis_attenuation_std_d = {}

folder_path = '..\\data\\public\\raw_data'  
for folder_name in os.listdir(folder_path):

    print(folder_name)
    if os.path.isdir(os.path.join(folder_path, folder_name)):
        for filename in os.listdir(os.path.join(folder_path, folder_name)):
            if filename.endswith('2model mask Analysis.txt'):
                with open(os.path.join(folder_path, folder_name, filename), 'r') as file:
                    lines = file.readlines()
                    thickness = None
                    thickness_std = None
                    surface_smoothness = None
                    surface_smoothness_std = None
                    edj_smoothness = None
                    edj_smoothness_std = None
                    epidermis_attenuation = None
                    epidermis_attenuation_std = None
                    dermis_attenuation = None
                    dermis_attenuation_std = None

                    x = 0
                    for line in lines:
                        if 'average epidermal thickness:' in line:
                            thickness = float(line.split(':')[1].strip().split('弮m')[0])
                            thickness_std = float(line.split('standard deviation(弮m):  ')[1].strip().split('弮m')[0])
                        elif 'difference average:' in line and x==0:
                            surface_smoothness = float(line.split(':')[1].strip().split('弮m')[0])
                            surface_smoothness_std = float(line.split('standard deviation(弮m):  ')[1].strip().split('弮m')[0])
                            x = x+1
                        elif 'difference average:' in line and x==1:
                            edj_smoothness = float(line.split(':')[1].strip().split('弮m')[0])
                            edj_smoothness_std = float(line.split('standard deviation(弮m):  ')[1].strip().split('弮m')[0])
                        elif 'average of epidermis:' in line:
                            epidermis_attenuation = float(line.split(':')[1].strip().split('(1/mm)')[0])
                            epidermis_attenuation_std = float(line.split('standard deviation(弮m):')[1].strip().split('(')[0])
                        elif 'average of dermis' in line:
                            dermis_attenuation = float(line.split(':')[1].strip().split('(1/mm)')[0])
                            dermis_attenuation_std = float(line.split('standard deviation(弮m):')[1].strip().split('(')[0])
                print(dermis_attenuation_std)
                thickness_d[folder_name]=thickness
                thickness_std_d[folder_name]=thickness_std
                surface_smoothness_d[folder_name]=surface_smoothness
                surface_smoothness_std_d[folder_name]=surface_smoothness_std
                edj_smoothness_d[folder_name]=edj_smoothness
                edj_smoothness_std_d[folder_name]=edj_smoothness_std
                epidermis_attenuation_d[folder_name]=epidermis_attenuation
                epidermis_attenuation_std_d[folder_name]=epidermis_attenuation_std
                dermis_attenuation_d[folder_name]=dermis_attenuation
                dermis_attenuation_std_d[folder_name]=dermis_attenuation_std



print(thickness_d)
import pickle



print(len(thickness_d))

with open(folder_path+'\\thickness_d.pkl', 'wb') as file:
    pickle.dump(thickness_d, file)
with open(folder_path+'\\thickness_std_d.pkl', 'wb') as file:
    pickle.dump(thickness_std_d, file)
with open(folder_path+'\\surface_smoothness_d.pkl', 'wb') as file:
    pickle.dump(surface_smoothness_d, file)
with open(folder_path+'\\surface_smoothness_std_d.pkl', 'wb') as file:
    pickle.dump(surface_smoothness_std_d, file)
with open(folder_path+'\\edj_smoothness_d.pkl', 'wb') as file:
    pickle.dump(edj_smoothness_d, file)
with open(folder_path+'\\edj_smoothness_std_d.pkl', 'wb') as file:
    pickle.dump(edj_smoothness_std_d, file)
with open(folder_path+'\\epidermis_attenuation_d.pkl', 'wb') as file:
    pickle.dump(epidermis_attenuation_d, file)
with open(folder_path+'\\epidermis_attenuation_std_d.pkl', 'wb') as file:
    pickle.dump(epidermis_attenuation_std_d, file)
with open(folder_path+'\\dermis_attenuation_d.pkl', 'wb') as file:
    pickle.dump(dermis_attenuation_d, file)
with open(folder_path+'\\dermis_attenuation_std_d.pkl', 'wb') as file:
    pickle.dump(dermis_attenuation_std_d, file)