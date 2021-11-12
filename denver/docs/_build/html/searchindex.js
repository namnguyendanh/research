Search.setIndex({docnames:["denver/denver","denver/denver.data","denver/denver.embeddings","denver/denver.learners","denver/denver.metrics","denver/denver.models","denver/denver.trainers","denver/denver.uncertainty_estimate","denver/denver.utils","etc/author","index","models/experiment_result","models/flair_seq_tagger","models/onenet","models/ulmfit_cls","tutorial/tutorial_hiperopt","tutorial/tutorial_ic","tutorial/tutorial_ner","tutorial/tutorial_onenet","user/cli","user/configs","user/installation","user/introduction","user/training_data"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.viewcode":1,nbsphinx:3,sphinx:56},filenames:["denver/denver.rst","denver/denver.data.rst","denver/denver.embeddings.rst","denver/denver.learners.rst","denver/denver.metrics.rst","denver/denver.models.rst","denver/denver.trainers.rst","denver/denver.uncertainty_estimate.rst","denver/denver.utils.rst","etc/author.rst","index.rst","models/experiment_result.rst","models/flair_seq_tagger.rst","models/onenet.rst","models/ulmfit_cls.rst","tutorial/tutorial_hiperopt.rst","tutorial/tutorial_ic.rst","tutorial/tutorial_ner.rst","tutorial/tutorial_onenet.rst","user/cli.rst","user/configs.rst","user/installation.rst","user/introduction.rst","user/training_data.rst"],objects:{"denver.data":{data_source:[1,0,0,"-"],dataset:[1,0,0,"-"],dataset_reader:[1,0,0,"-"],preprocess:[1,0,0,"-"]},"denver.data.data_source":{DenverDataSource:[1,1,1,""]},"denver.data.data_source.DenverDataSource":{build_corpus:[1,2,1,""],build_databunch:[1,2,1,""],convert_to_bio_file:[1,2,1,""],from_csv:[1,2,1,""],from_df:[1,2,1,""]},"denver.data.dataset":{DenverDataset:[1,1,1,""]},"denver.data.dataset.DenverDataset":{get_labels:[1,2,1,""],get_sentences:[1,2,1,""],normalize_df:[1,2,1,""]},"denver.data.dataset_reader":{OneNetDatasetReader:[1,1,1,""]},"denver.data.dataset_reader.OneNetDatasetReader":{get_spans:[1,2,1,""],text_to_instance:[1,2,1,""]},"denver.data.preprocess":{BalanceLearn:[1,1,1,""],normalize:[1,3,1,""],split_data:[1,3,1,""],standardize_df:[1,3,1,""]},"denver.data.preprocess.BalanceLearn":{subtext_sampling:[1,2,1,""]},"denver.embeddings":{embeddings:[2,0,0,"-"]},"denver.embeddings.embeddings":{Embeddings:[2,1,1,""]},"denver.embeddings.embeddings.Embeddings":{embed:[2,2,1,""],fine_tuning:[2,2,1,""]},"denver.learners":{base_learner:[3,0,0,"-"],flair_sequence_tagger_leaner:[3,0,0,"-"],onenet_learner:[3,0,0,"-"],ulmfit_cls_learner:[3,0,0,"-"]},"denver.learners.base_learner":{DenverLearner:[3,1,1,""]},"denver.learners.base_learner.DenverLearner":{evaluate:[3,2,1,""],predict:[3,2,1,""],process:[3,2,1,""],save_model:[3,2,1,""],train:[3,2,1,""]},"denver.learners.flair_sequence_tagger_leaner":{FlairSequenceTaggerLearner:[3,1,1,""]},"denver.learners.flair_sequence_tagger_leaner.FlairSequenceTaggerLearner":{convert_to_rasa_format:[3,2,1,""],evaluate:[3,2,1,""],predict:[3,2,1,""],predict_on_df:[3,2,1,""],process:[3,2,1,""],train:[3,2,1,""],validate:[3,2,1,""]},"denver.learners.onenet_learner":{OnenetLearner:[3,1,1,""]},"denver.learners.onenet_learner.OnenetLearner":{convert_to_rasa_format:[3,2,1,""],evaluate:[3,2,1,""],get_classes:[3,2,1,""],predict:[3,2,1,""],predict_on_df:[3,2,1,""],process:[3,2,1,""],train:[3,2,1,""],validate:[3,2,1,""]},"denver.learners.ulmfit_cls_learner":{ULMFITClassificationLearner:[3,1,1,""]},"denver.learners.ulmfit_cls_learner.ULMFITClassificationLearner":{convert_to_rasa_format:[3,2,1,""],evaluate:[3,2,1,""],evaluate_by_step:[3,2,1,""],fit:[3,2,1,""],freeze:[3,2,1,""],freeze_to:[3,2,1,""],get_classes:[3,2,1,""],get_uncertainty_score:[3,2,1,""],predict:[3,2,1,""],predict_on_df:[3,2,1,""],predict_on_df_by_step:[3,2,1,""],predict_with_mc_dropout:[3,2,1,""],process:[3,2,1,""],save:[3,2,1,""],train:[3,2,1,""],unfreeze:[3,2,1,""],validate:[3,2,1,""]},"denver.metrics":{Metrics:[4,1,1,""],OnetNetMetrics:[4,1,1,""],get_metric:[4,3,1,""]},"denver.metrics.Metrics":{accuracy:[4,2,1,""],add_fn:[4,2,1,""],add_fp:[4,2,1,""],add_tn:[4,2,1,""],add_tp:[4,2,1,""],f_score:[4,2,1,""],get_classes:[4,2,1,""],get_fn:[4,2,1,""],get_fp:[4,2,1,""],get_tn:[4,2,1,""],get_tp:[4,2,1,""],macro_avg_accuracy:[4,2,1,""],macro_avg_f_score:[4,2,1,""],macro_avg_precision:[4,2,1,""],macro_avg_recall:[4,2,1,""],micro_avg_accuracy:[4,2,1,""],micro_avg_f_score:[4,2,1,""],micro_avg_precision:[4,2,1,""],micro_avg_recall:[4,2,1,""],precision:[4,2,1,""],recall:[4,2,1,""]},"denver.metrics.OnetNetMetrics":{get_metric:[4,2,1,""],reset:[4,2,1,""]},"denver.models":{onenet:[5,0,0,"-"]},"denver.models.onenet":{OneNet:[5,1,1,""]},"denver.models.onenet.OneNet":{decode:[5,2,1,""],forward:[5,2,1,""],get_metrics:[5,2,1,""],get_predicted_tags:[5,2,1,""],get_spans:[5,2,1,""]},"denver.trainers":{language_model_trainer:[6,0,0,"-"],trainer:[6,0,0,"-"]},"denver.trainers.language_model_trainer":{LanguageModelTrainer:[6,1,1,""]},"denver.trainers.language_model_trainer.LanguageModelTrainer":{downdload_lm:[6,2,1,""],fine_tuning_from_df:[6,2,1,""],fine_tuning_from_folder:[6,2,1,""],fine_tuning_from_wiki:[6,2,1,""],get_wiki:[6,2,1,""],split_wiki:[6,2,1,""]},"denver.trainers.trainer":{ModelTrainer:[6,1,1,""]},"denver.trainers.trainer.ModelTrainer":{train:[6,2,1,""]},"denver.uncertainty_estimate":{UncertaintyEstimator:[7,1,1,""]},"denver.uncertainty_estimate.UncertaintyEstimator":{entropy:[7,2,1,""],get_uncertainty_score:[7,2,1,""]},"denver.utils":{config_parser:[8,0,0,"-"],utils:[8,0,0,"-"]},"denver.utils.config_parser":{ConfigParserMultiValues:[8,1,1,""],get_config_section:[8,3,1,""],get_config_yaml:[8,3,1,""]},"denver.utils.config_parser.ConfigParserMultiValues":{getlist:[8,2,1,""]},"denver.utils.utils":{check_url_exists:[8,3,1,""],convert_to_BIO:[8,3,1,""],convert_to_denver_format:[8,3,1,""],convert_to_ner:[8,3,1,""],download_url:[8,3,1,""],ifnone:[8,3,1,""],load_json:[8,3,1,""],rename_file:[8,3,1,""],tokenize:[8,3,1,""]},denver:{data:[1,0,0,"-"],embeddings:[2,0,0,"-"],learners:[3,0,0,"-"],metrics:[4,0,0,"-"],models:[5,0,0,"-"],trainers:[6,0,0,"-"],uncertainty_estimate:[7,0,0,"-"],utils:[8,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"001":[3,6,18,20],"0575":6,"100":2,"100d":20,"1024":[2,3,17,20],"123":1,"128":[1,3,6,16,18,20],"150":18,"200":[3,18,20],"2018":12,"2019":12,"2020":21,"2021":9,"2048":[17,20],"250":2,"27th":12,"280m":11,"2b0":[15,16,17,18],"300":[17,20],"30x35x":8,"500":[2,3],"50d":[3,18,20],"517m":11,"520m":11,"75854855":[15,16,17,18],"82m":11,"8765e803":19,"8820":11,"9162":11,"92619":11,"9320":11,"9343":11,"9498":11,"9619":11,"9762":11,"9794":11,"9803":11,"9994359612464905":3,"9999693632125854":8,"9999999":3,"\u0103n":17,"abstract":[3,22],"b\u1eb1ng":16,"boolean":5,"c\u00f2n":3,"c\u0169i":3,"ch\u1ea5t":16,"char":20,"class":[1,2,3,4,5,6,7,8,20],"default":[2,5,6,19],"export":[3,19],"final":[3,6,20],"float":[1,2,3,4,5,6],"function":[1,2,3,4,6,8,20],"gh\u1ebf":17,"import":[16,17,18],"int":[1,2,3,6,20],"l\u00e0m":16,"li\u1ec7u":16,"n\u00e0y":3,"new":[3,5,20],"public":11,"return":[1,2,3,4,5,7,8,21],"static":8,"true":[1,2,3,5,6,7,8,16,17,18,20],"try":21,"v\u00e0ng":18,"v\u1eady":16,"while":22,For:[9,16,17,18,20,23],IDs:22,The:[1,2,3,6,7,8,16,17,19,20,22,23],Their:22,Then:21,Use:[16,17,18],Used:5,With:[20,23],__name__:[3,7],abl:22,about:[10,11,22],abov:[16,22,23],acc:[3,4],accordingli:22,accumul:[4,5],accuraci:4,activ:[20,21],actual:5,adam:3,add:[5,20],add_fn:4,add_fp:4,add_tn:4,add_tp:4,addit:[16,17,18],addition:[4,20],adopt:22,advantag:22,after:[1,3,8,16,17,18],aim:22,akbik:12,alan:12,algebra:22,all:[3,4,22],allennlp:[1,4,5],allow:[1,10,19,22],almost:22,alongsid:22,also:[5,10,16,17,18,19],american:12,amount:22,anaconda3:21,anaconda:21,analyt:22,ani:[4,6,8,9,15,16,17,18,21],anneal:2,annual:12,api:10,applic:22,approach:22,appropri:20,architectur:[2,6,10],arg:20,argmax:5,argument:[8,20],artifici:22,ask_confirm:8,assembl:22,associ:[12,22],ati:10,audienc:22,augment:22,author:10,automat:22,avail:[10,22],ave:21,averag:[3,20],awd_lstm:6,babe:[6,16,17,20],backward:20,balac:20,balanc:[1,20],balancelearn:1,base:[1,2,3,4,5,6,7,8,20,22],base_learn:[0,10],base_path:[3,6,16,18,20],basic:[10,22],batch:[1,2,3,6,7,16,20],batch_siz:[1,2,3,6,16,17,18,20],becaus:[1,22],been:22,befor:[3,16],behind:22,below:19,bergmann:12,best:[3,6,20,22],beta:[3,4,20],between:5,bidirect:[3,18,20],bigger:22,binari:20,bio:[1,8],block:22,bloom:22,blyth:12,bool:[1,2,3,6,8],boost:22,both:22,broader:22,broken:1,build:[1,10,22],build_corpu:1,build_databunch:1,built:[10,20],bum:13,busi:22,calcul:[3,5,7,22],calculate_span_f1:5,call:[5,22,23],can:[16,17,18,19,20,22],certain:22,chang:8,chao:19,chapter:12,char_embedding_dim:[3,18,20],char_encoder_typ:[3,18,20],charact:[1,3,20,22],characterist:22,chart:3,cheat:[19,20],check:[8,10],check_url_exist:8,checkpoint:[2,17],chunk:1,claim:22,class_nam:4,classif:[3,10,14,16,22],classifi:[1,4,16,20],classmethod:1,clean_lm:3,cli:[19,20],client:22,clone:21,cnn:[3,18,20],code:[10,22],cole:12,collect:[8,22],column:[1,3,20,23],com:[9,21],combin:22,come:22,comet:20,cometv3:[10,18,20],command:10,comment:9,commerci:22,common:22,compat:5,compon:[10,20],compos:22,composit:22,comprehens:5,compris:20,comput:[5,12,22],con:[18,23],conda:21,confer:12,confid:[3,8],config:[8,19,20],config_fil:[8,19],config_pars:[0,10],configparsermultivalu:8,configur:[8,10,19,22],confus:3,confusion_matrix:3,confust:3,consist:[3,19],constrain_crf_decod:5,consum:22,contact:9,contain:4,contextu:12,contrain:8,conv_layer_activ:[3,18,20],convert:[1,3,5,8],convert_to_bio:8,convert_to_bio_fil:1,convert_to_denver_format:8,convert_to_n:8,convert_to_rasa_format:3,convolut:20,copi:19,copyright:9,coresspod:19,corpu:[1,2,16,17,20],corpus_dir:[2,17],correct:4,correspond:23,corrrespond:20,cours:22,creat:[16,17,18,21],crf:20,crf_decod:5,csv:[1,3,16,17,18,19,20,23],current:3,dai:18,data:[0,2,3,6,8,10,16,17,18,19,20,22],data_bab:16,data_df:[1,6,8,16,17,18],data_fil:19,data_fold:[6,16],data_fp:8,data_path:18,data_sourc:[0,3,10,16,17,18],data_split_1:16,data_split_2:16,data_split_n:16,databunch:1,datafram:[1,3,6,8,16,17,18,20],dataset:[0,3,6,10,16,17,18,19,20,22],dataset_read:[0,10],datasetread:1,declar:22,decod:[5,20,22],deep:[10,22],defalt:7,defin:[16,17,18,19,20,22],definit:[20,22],deliv:22,deliveri:22,democrat:22,demonstr:22,dender:21,denver:[9,15,16,17,18,19,20,22,23],denverdataload:1,denverdataset:1,denverdatasourc:[1,3,16,17,18],denverlearn:3,depend:20,descript:20,dest:8,detail:19,detailed_result:3,dev:[3,15,16,17,18,19],dev_eval_result:3,dev_loss:3,develop:[9,16,17,22],devic:3,dialogu:22,dict:[3,4,5,7,8],dictionari:[2,5,8,20],differ:22,difficult:22,dim:20,dimens:20,directli:20,directori:[3,6,8,20],disabl:[3,6],disallow:1,dist:21,doe:5,domain:[13,22],don:1,down:1,downdload_lm:6,download:[6,8,19,21],download_url:8,downstream:3,drop_mult:[3,6,16,20],dropout:[2,3,5,6,17,18,20],duncan:12,dure:[3,5,6,20],each:[1,2,5,6,16,20],earli:[3,5],easi:[10,22],easier:22,efd:22,effect:[19,20,22],effici:22,either:22,els:[3,6,20],email:9,emb:[2,17],embed:[0,3,10,12,17],embedding_s:2,embedding_typ:[2,17,20],emerg:22,emoiji:20,emoji:[1,3,20],empti:5,enabl:22,encapsul:22,encod:[5,16,17,20,22],encourag:22,end:[3,6,8],engin:22,english:20,entir:3,entiti:[3,8,10,12,17,22],entropi:7,enviro:21,environ:10,epoch:[2,3,5,6,20],equal:23,estim:[4,7],etc:10,eval:3,evalu:[3,4,6,10,20],evaluate_by_step:3,evalut:[3,16,17,18,19],everi:[3,22],evolut:22,exampl:[3,6,8,10,16,17,18,19,22,23],exist:[8,15,16,17,18,21],exit:19,exp_lr:6,experi:[6,10,22],experiemnt:11,experiment:22,expert:22,explain:[16,17],expos:22,exposur:22,extens:[19,22],extractor:[3,8],f_score:4,face:22,fals:[1,2,3,4,5,6,7,8,16,17,20],faq:22,far:[16,17],fashion:1,fasl:20,fast:22,fastai:[1,3],favor:22,featur:22,feedforward:5,figur:3,file:[1,3,6,8,10,16,17,18,19,22,23],file_path:8,filenam:8,filter:20,find:[19,20],fine:[2,6,14,16,17],fine_tun:[2,17],fine_tuning_from_df:[6,16],fine_tuning_from_fold:[6,16],fine_tuning_from_wiki:6,first:22,fit:3,flair:[1,2,3,19],flair_config:19,flair_sequence_tagger_lean:[0,10],flairsequencetagg:[3,8,10,11,16,18],flairsequencetaggerlearn:[3,17],flexibl:22,focal_loss_gamma:5,focu:22,folder:[2,6,8,17,19],follow:[1,4,7,8,9,16,17,20,21],format:[1,3,7,8,10,16,17,19,20],forward:[2,5,17,20],frame:1,freez:3,freeze_to:3,frequent:5,from:[1,3,8,10,16,17,18,19,21,22],from_csv:[1,16,17,18],from_df:[1,16,17,18],ftech:[15,16,17,18,19,22],full:22,fundament:22,fusion:22,gener:[20,22],get:[1,3,4,7,8,10],get_class:[3,4],get_config_sect:8,get_config_yaml:8,get_fn:4,get_fp:4,get_label:1,get_metr:[4,5],get_predicted_tag:5,get_sent:1,get_span:[1,5],get_tn:4,get_tp:4,get_uncertainty_scor:[3,7,16],get_wiki:6,getlist:8,git:21,github:21,given:[3,16,17,18],glove:[3,18],gmail:9,got:22,gpu:22,grad_norm:3,gradient:[3,22],greatli:22,ground:4,group:3,grow:22,handl:[5,22],happen:22,has:[20,22],have:[1,3,5,16,17,20,22],head:18,help:19,here:[1,5,16,17,18,19,20,21],hidden:[2,20],hidden_s:[2,3,17,18,20],high:22,highli:22,hiperopt:10,how:[16,17,22],howard:14,howev:22,http:[15,16,17,18,19],hyper:[19,20],hyper_param:20,ic_config:19,iclass:[3,7],idea:22,ids:5,ifnon:8,illustr:[16,17,18],imag:[3,22],implement:[3,5,22],improv:[2,3],includ:[1,3,4,10,16,20,22,23],include_start_end_transit:5,increas:22,index:[10,16,17],indic:5,industri:22,inexperienc:22,infer:[3,10,11,16,17,18],inform:22,inherit:22,ini:8,initi:[3,5,6],initializerappl:5,innov:22,input:[1,8,19,20,22,23],input_featur:20,instal:[10,15,16,17,18],instanc:1,instanti:22,instead:18,instruct:[19,21],integ:22,intent:[1,3,5,7,10,13,16,18,20,22,23],intent_col:[1,3,18,20],intent_label:5,intent_label_namespac:5,interest:9,interfac:[10,22],intern:[4,12,22],introduc:22,introduct:10,involv:22,is_forward_lm:2,is_norm:17,is_save_best_model:[3,6,20],is_stratifi:1,item:1,its:[19,20],jeremi:14,joint:[13,20],jointidsf:11,jointli:[1,18],json:8,just:[21,22],karl:13,kei:[4,5,8],kera:22,kim:13,kwarg:[1,3,5,6],label:[1,3,4,5,12,16,17,18,20,23],label_col:[1,3,16,17,18,20],label_encod:5,lang:6,langu:16,languag:[2,6,10,13,14,16,17,22],language_model_train:[0,10,16],languagemodeltrain:[6,16],larger:1,last:[3,22],layer:[2,3,20],lazi:1,lead:22,learn:[2,3,6,10,16,17,18,20,22],learner:[0,6,10,16,17,18],learning_r:[2,3,6,16,17,18,20],lee:13,length:2,let:22,level:[20,22],librari:[10,15,16,17,18,21,22],like:[16,17,18,20],limit:22,line:[1,10],linear:20,linguist:12,link:19,list:[1,3,4,5,6,8,16,20,22],lm_fn:[6,16],lm_fns_path:16,lm_trainer:16,load:[16,17],load_json:8,lock:20,locked_dropout:[3,17,20],log:[3,20],lookup:10,loss:3,loss_func:20,lower:20,lowercas:[1,3,16,17,18,20],lowercase_token:20,lstm:[3,18,20],machin:22,macro:20,macro_avg_accuraci:4,macro_avg_f_scor:4,macro_avg_precis:4,macro_avg_recal:4,mai:19,main:[3,10,11,19,22],main_scor:3,mainli:22,make:[19,22],map:[20,22],massiv:22,matric:3,matrix:3,mau:18,max:2,max_epoch:[2,17],maximum:3,mayb:[3,16,17,18,20],measur:4,member:22,mention:22,messag:19,metadata:5,method:[3,5,7,16],metic:4,metric:[0,3,5,10,16,17,18],micro:20,micro_avg_accuraci:4,micro_avg_f_scor:4,micro_avg_precis:4,micro_avg_recal:4,min:2,mini:1,mini_batch_chunk_s:1,minio:[15,16,17,18,19],mode:[3,4,5,16,17,18],model:[0,2,3,4,6,7,8,10,11,14,22],model_dir:[2,17,19],model_fil:[3,6,16,17,18,20],model_path:[3,16,17,18],modeltrain:[6,16,17,18],modul:[10,22],modular:22,mom:[3,6,16],momentum:[3,6,20],monitor_test:6,more:[1,22],most:22,move:22,much:22,multi:20,multipl:[6,20],must:[3,16,19,23],n_time:[3,7,16],nam:19,name:[1,2,3,4,6,8,10,12,17,19,20,21,22],natur:22,need:[3,10,16,17,18,19,20,22],ner:[1,3,10,11,18,19,20,23],ner_config:19,nest:5,network:22,neural:22,ngram:20,ngram_filter_s:[3,18,20],nlayer:2,nlp:22,nlu:1,none:[1,2,3,4,5,6,8,15,16,17,18,20,21],norm:3,normal:1,normalize_df:1,north:12,note:19,novel:22,now:22,num_epoch:[3,6,16,17,18,20],num_filt:[3,18,20],num_lay:[3,20],num_wok:6,num_work:[1,6],number:[1,2,3,6,7,20,23],numer:22,object:[1,2,3,4,5,6,7],object_typ:[3,23],observ:22,obtain:22,odel:22,old:[3,6,8],onc:1,one:22,onenet:[0,4,10,11,19],onenet_config:19,onenet_learn:[0,10],onenetdatasetread:1,onenetlearn:[3,18],onetnet:3,onetnetmetr:4,onli:[16,17],optim:[3,19],option:[1,3,5,6,16,19],order:[5,10],ordereddict:8,organ:21,other:[8,16,17,20,22],otherwis:[8,20],out_fil:[1,16],outfil:17,output:[3,16,17,18,20,22],output_dict:5,output_featur:20,outspread:22,over:[5,22],overal:4,overwrit:[3,6,8],packag:[10,21],page:[10,16,17],parallel:22,param:[8,20],paramet:[1,2,3,4,5,6,7,8,19,20],pars:8,particular:22,pass:[5,16,19],path:[1,2,3,6,8,16,17,18,19,20],patienc:[2,3],patient:3,pct:[1,6,19],penalti:5,per:4,percent:1,perform:[19,22],phan:9,phanxuanphucnd:[9,21],phuc:9,piec:22,pip:[15,16,17,18,21],pipelin:22,pkl:[16,20],pleas:9,point:[7,22],pool:[2,12],pooled_flair_embed:[2,17,20],popul:5,posit:5,possibl:22,pprint:[16,17,18],practic:22,practition:22,pre:[1,17,20],pre_process:20,precis:[3,4],pred:3,predction:7,predict:[3,7,10,13],predict_on_df:[3,16,17,18],predict_on_df_by_step:[3,16],predict_with_mc_dropout:3,prepar:[10,22],preprocess:[0,10,22],present:22,pretokenis:1,pretrain:[2,6,10,16,17],pretrain_embed:20,pretrain_language_model:20,primit:22,print:17,prob:7,probabl:[2,3,7,20],problem:[20,22],process:[1,3,16,17,18,20,22],produc:22,product:22,programat:10,programmat:10,project:20,prove:22,provid:[3,4,5,8,10,16,17,19,20,22],punctuat:[1,3,20],purpos:1,py3:[15,16,17,18,21],python:[10,21],pytorch:[10,22],query_kb:[3,23],question:9,quickli:20,rais:5,rasa:[3,8,16,17],rate:[2,3,6,20],rather:[5,22],ratio:[1,6,19],raw:8,read:1,readi:22,realli:22,recal:[3,4],recent:22,recognit:[10,12,17,22],recommend:22,refer:[12,13,14,16,17,18,19,22],regard:22,regular:5,regularizerappl:5,relat:22,reliabl:22,relu:[3,18,20],remov:20,rename_fil:8,replac:[1,3,16,20],replic:22,report:6,repositori:21,reproject:20,reproject_embed:[3,17,20],requir:[5,19,22],rescal:3,research:22,reset:[4,5],resiz:22,resourc:[15,16,17,18,19],result:[3,5,6,7,10],retrospect:22,retun:3,reusabl:22,rm_emoiji:20,rm_emoji:[1,3,16,20],rm_special_token:[1,3,16,20],rm_url:[1,3,16,20],rnn:[2,20],rnn_layer:[2,3,17,20],rnn_type:[3,18,20],roland:12,row:1,ruder:14,run:21,sale:22,salebot:[1,16],same:[1,16,20,22],sampel:3,sampl:[1,3,7,16,17,18,19],satisfi:22,save:[1,2,3,6,8,20],save_cm_dir:3,save_dir:3,save_fil:3,save_model:3,save_nam:3,scale:22,score:[3,7,11,16,20],scratch:22,search:10,sebastian:14,second:22,section:8,see:[19,21],seed:[1,6],select:7,sentenc:[1,17,20,22,23],separ:[16,17,18,20],sequenc:[1,2,3,12],sequence_length:2,sequence_logit:5,serial:5,sestion:8,set:[1,16,17,18],setup:20,sheet:[19,20],shop:[8,16,17],should:[5,22],show:[6,19],shuffl:1,similar:[19,22],simpl:[5,10],simplifi:22,size:[1,2,3,6,7,8,11,20],skip_save_log:3,skip_save_model:[3,6],slot:13,softmax:7,softwar:22,some:[5,20,22],someth:22,sophist:22,sota:[10,19],sourc:[1,2,3,4,5,6,7,8,16,17,18],space:[1,3,20],span:4,span_tag:1,special:[1,3,20],specif:20,specifi:1,speech:22,spend:22,split:[1,6,19],split_data:1,split_wiki:6,spoken:13,standard:[1,22,23],standardize_df:1,start:[3,8,16],state:[2,4,5,20],step:[16,17],stop:[3,5],storag:[3,19,20],store:6,str:[1,2,3,5,6,8,20],stratifi:1,strato:13,string:[5,12],structur:[17,20],sub:[6,22],subtext_sampl:1,successfulli:21,suitabl:1,sungjin:13,system:8,tab:19,tabl:23,tag:[1,3,5,17,18,20,23],tag_col:[1,3,18,20],tag_id:5,tag_label_namespac:5,tag_typ:[3,17],tagger:3,tags_col:3,take:[1,8,22],taken:19,tanja:12,tar:[18,20],target:4,task:[1,3,10,16,17,18,22,23],team:22,technolog:22,temp_output:1,tempor:1,ten:22,tensor:[5,7,22],tensorflow:22,term:22,test:[1,2,3,6,10,16,17,18,19,20],test_df:[1,16,17,18],test_path:[1,16,17,18,19,20],test_siz:19,text:[1,3,7,8,14,16,17,18,19,20,22,23],text_col:[1,3,16,17,18,20],text_field_embedd:5,text_input:19,text_to_inst:1,than:[1,5,22],thank:9,theano:22,thei:1,them:[20,22],thi:[1,3,5,8,10,16,17,19],those:22,time:[3,7,11,21,22],to_csv:[16,17],togeth:22,token:[1,3,5,8,20,22],token_delimit:1,token_index:1,toolbox:[10,22],top:20,train:[1,2,3,4,5,6,10,11,20,22],train_df:[1,16,17,18],train_path:[1,16,17,18,20],train_split_1:[2,17],train_split_2:[2,17],train_split_x:[2,17],trainabl:20,trainer:[0,2,5,10,16,17,18],training_model:20,transform:22,truth:4,tun:6,tune:[2,6,14,16,17],tupl:6,tutori:10,two:[10,22],txt:[1,2,8,16,17],type:[1,3,4,5,8,20,22],ugli:5,ulmfit:[3,19],ulmfit_cls_learn:[0,10],ulmfit_config:19,ulmfitclassifi:[10,11,17,18],ulmfitclassificationlearn:[3,16],uncas:17,uncertainti:[3,7,16],uncertainty_estim:[0,10],uncertainty_scor:[3,7,16],uncertaintyestim:7,unchang:22,understand:[10,13,22],unfreez:3,unifi:22,uninstal:[15,16,17,18,21],union:[1,2,3,6,8],univers:14,until:[2,5],url:[1,3,6,8,20],use:[2,3,5,10,16,19,20,21,22],use_crf:[3,17,20],use_rnn:[3,20],use_tensorboard:6,used:[5,11,16,19,20,21],user:[8,10,20,22],using:[1,3,7,20,22],utf:[16,17],util:[0,10],valid:[1,2,3,6,17],valu:[1,3,4,7,8,20,22],varieti:22,vast:22,vector:22,verbos:[6,7],verbose_metr:5,veri:22,vi_wt_bab:16,vi_wt_vocab_bab:16,vicl:16,viclass:20,viet:19,vietnames:[1,3,20],view:22,viner:[17,20],vision:22,vocab:5,vocabulari:22,vollgraf:12,wait:2,want:19,wave:22,weight:[3,20],well:22,wellcom:22,were:22,what:10,when:[2,3,6,22],where:[1,22],which:[4,5,22],whl:[15,16,17,18,21],why:10,wide:22,wiki:[6,16,20],wikipedia:6,wise:5,with_confusion_matrix:3,with_dropout:[3,16],with_uncertainti:3,with_uncertainty_scor:3,without:[1,2,3,10,20,22],word:[1,5,22,23],word_dropout:[3,17,20],word_embedding_dim:[3,18,20],word_pretrained_embed:[3,18,20],work:22,worker:6,workhors:22,write:[10,21,22],xin:19,xxx_embed:20,y_pred:4,y_true:4,yaml:19,year:22,you:[9,16,17,18,19,20,21],young:13,your:[9,10]},titles:["Denver Package","denver.data","denver.embeddings","denver.learners","denver.metrics","denver.models","denver.trainers","denver.uncertainty_estimate","denver.utils","About the Author","Welcome to Denver\u2019s Documentation!","Experiment Results","FlairSequenceTagger","OneNet","ULMFITClassifier","Tutorial: Use hiperopt","Tutorial: Building IC models","Tutorial: Building NER models","Tutorial: Building OneNet model","Command Line Interface","Configuration File","Installation","Introduction","Training Data Format"],titleterms:{Use:15,about:9,architectur:22,argument:19,ati:11,author:9,base_learn:3,basic:20,build:[16,17,18],built:22,check:21,cometv3:11,command:19,config_pars:8,configur:20,content:[11,15,16,17,18,19,20,21,22],data:[1,23],data_sourc:1,dataset:[1,11],dataset_read:1,denver:[0,1,2,3,4,5,6,7,8,10,21],document:10,embed:[2,20],environ:21,evalu:[16,17,18,19],exampl:20,experi:[11,19],file:20,flair:20,flair_sequence_tagger_lean:3,flairsequencetagg:[12,17,20],format:23,get:[16,17,18,19],glove:20,hiperopt:[15,19],indic:10,infer:19,instal:21,interfac:19,introduct:22,languag:20,language_model_train:6,learner:3,line:19,lookup:20,metric:4,model:[5,16,17,18,19,20],ner:17,note:[16,17,18,23],onenet:[5,13,18,20],onenet_learn:3,packag:0,predict:[16,17,18,19],prepar:21,preprocess:1,pretrain:20,result:11,tabl:[10,11,15,16,17,18,19,20,21,22],thi:22,train:[16,17,18,19,23],trainer:6,tutori:[15,16,17,18],ulmfit_cls_learn:3,ulmfitclassifi:[14,16,20],uncertainty_estim:7,util:8,welcom:10,what:22,why:22,word:20,your:21}})