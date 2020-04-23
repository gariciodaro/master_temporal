
#Requiered libs
import sys
import os
import pandas as pd
import wget

#add path to OFHandlers.py (Object file handlers)
sys.path.append("/home/gari/Desktop/master_tesis_v3/")

#import Object file handler
#I build this script to easy the call
#to pickle save and load from disk.
from OFHandlers import OFHandlers as OFH

#path to csv files of Bio bank database
path_to_data="/home/gari/Desktop/master_tesis_v3/Data/"

indi_inattentive=OFH.load_object(path_to_data+"BioBankInfo/indi_inattentive.file")
indi_hyperactive=OFH.load_object(path_to_data+"BioBankInfo/indi_hyperactive.file")
indi_combined=OFH.load_object(path_to_data+"BioBankInfo/indi_combined.file")
indi_healthy=OFH.load_object(path_to_data+"BioBankInfo/indi_healthy.file")



def download_patients(list_subject,type_sub,path_to_store):
	#good_subjects=[]
	for each_subject in list_subject:
		print('Beginning download of subject:',each_subject)
		# Amazon web service url. Biobank data.
		url_aws='https://fcp-indi.s3.amazonaws.com/data/Projects/HBN/EEG/'
		try:
			#store the raw file on external drive!
			folder_to_save=path_to_store+type_sub+'/'+each_subject

			#create the folder to hold subject data
			os.mkdir(folder_to_save)
			url_1=url_aws+each_subject+"/EEG/preprocessed/csv_format/RestingState_chanlocs.csv"
			wget.download(url_1, folder_to_save+"/RestingState_chanlocs.csv")
			
			url_2=url_aws+each_subject+"/EEG/preprocessed/csv_format/RestingState_data.csv"
			print("url_2",url_2)
			wget.download(url_2, folder_to_save+"/RestingState_data.csv")
			
			url_3=url_aws+each_subject+"/EEG/preprocessed/csv_format/RestingState_event.csv"
			wget.download(url_3, folder_to_save+"/RestingState_event.csv")
			good_subjects.append(each_subject)
			print('Finished correctly download of subject:',each_subject)
		except:
			#delete the folder of subject if download fails
			try:
				os.rmdir(folder_to_save)
			except:
				pass
	
	#OFH.save_object(path_to_store+'TrackSubjects/'+type_sub+'_good.file',good_subjects)

if __name__== "__main__":

	#this folder should contain the folowing structure
	# /media/gari/extra_ssd/RawBiobankData/inattentive
	# /media/gari/extra_ssd/RawBiobankData/hyperactive
	# /media/gari/extra_ssd/RawBiobankData/combined
	# /media/gari/extra_ssd/RawBiobankData/healthy
	# those folders should be empty before runnig this script
	path_to_store='/media/gari/extra_ssd/RawBiobankData/'


	print("inattentive")
	download_patients(indi_inattentive,"inattentive",path_to_store)

	print("hyperactive")
	download_patients(indi_hyperactive,"hyperactive",path_to_store)

	print("combined")
	download_patients(indi_combined,"combined",path_to_store)

	print("healthy")
	download_patients(indi_healthy,"healthy",path_to_store)
	