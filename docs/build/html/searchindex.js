Search.setIndex({docnames:["getting_started","index","installation","k-seq_package","k_seq.data","k_seq.estimate","k_seq.model","k_seq.utility"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,sphinx:56},filenames:["getting_started.rst","index.rst","installation.rst","k-seq_package.rst","k_seq.data.rst","k_seq.estimate.rst","k_seq.model.rst","k_seq.utility.rst"],objects:{"k_seq.data":{grouper:[4,0,0,"-"],preprocess:[4,0,0,"-"],seq_data:[4,0,0,"-"],seq_data_analyzer:[4,0,0,"-"],simu:[4,0,0,"-"],variant_pool:[4,0,0,"-"]},"k_seq.data.grouper":{Grouper:[4,1,1,""],GrouperCollection:[4,1,1,""],get_group:[4,4,1,""]},"k_seq.data.grouper.Grouper":{__init__:[4,2,1,""],axis:[4,3,1,""],get_table:[4,2,1,""],group:[4,3,1,""],split:[4,2,1,""],target:[4,3,1,""],type:[4,3,1,""]},"k_seq.data.grouper.GrouperCollection":{add:[4,2,1,""]},"k_seq.data.preprocess":{fastq_to_count:[4,4,1,""],load_Seqtable_from_count_files:[4,4,1,""],read_count_file:[4,4,1,""]},"k_seq.data.seq_data":{SeqData:[4,1,1,""],SeqTable:[4,1,1,""],slice_table:[4,4,1,""]},"k_seq.data.seq_data.SeqData":{__init__:[4,2,1,""],add_grouper:[4,2,1,""],add_sample_total:[4,2,1,""],add_spike_in:[4,2,1,""],from_count_files:[4,2,1,""],from_json:[4,2,1,""],from_pickle:[4,2,1,""],samples:[4,2,1,""],seq_table:[4,3,1,""],seqs:[4,2,1,""],to_json:[4,2,1,""],to_pickle:[4,2,1,""],update_analysis:[4,2,1,""]},"k_seq.data.seq_data.SeqTable":{__init__:[4,2,1,""],about:[4,2,1,""],density:[4,2,1,""],describe:[4,2,1,""],filter_axis:[4,2,1,""],samples:[4,2,1,""],seqs:[4,2,1,""],update_analysis:[4,2,1,""]},"k_seq.data.seq_data_analyzer":{SeqDataAnalyzer:[4,1,1,""],SeqTableAnalyzer:[4,1,1,""],cross_table_compare:[4,4,1,""],rep_variance_scatter:[4,4,1,""],sample_entropy_scatterplot:[4,4,1,""],sample_info:[4,4,1,""],sample_overview:[4,4,1,""],sample_overview_plots:[4,4,1,""],sample_rel_abun_hist:[4,4,1,""],sample_spike_in_ratio_scatterplot:[4,4,1,""],sample_total_reads_barplot:[4,4,1,""],sample_unique_seqs_barplot:[4,4,1,""],seq_length_dist:[4,4,1,""],seq_mean_value_detected_samples_scatterplot:[4,4,1,""],seq_overview:[4,4,1,""],seq_variance:[4,4,1,""]},"k_seq.data.simu":{DistGenerators:[4,1,1,""],PoolParamGenerator:[4,1,1,""],SimulationResults:[4,1,1,""],get_pct_gaussian_error:[4,4,1,""],simulate_counts:[4,4,1,""],simulate_on_byo_doped_condition_from_exp_results:[4,4,1,""],simulate_w_byo_doped_condition_from_param_dist:[4,4,1,""]},"k_seq.data.simu.DistGenerators":{compo_lognormal:[4,2,1,""],lognormal:[4,2,1,""],uniform:[4,2,1,""]},"k_seq.data.simu.PoolParamGenerator":{sample_from_dataframe:[4,2,1,""],sample_from_iid_dist:[4,2,1,""]},"k_seq.data.simu.SimulationResults":{__init__:[4,2,1,""],get_est_results:[4,2,1,""],get_fold_range:[4,2,1,""],get_uncertainty_accuracy:[4,2,1,""]},"k_seq.data.variant_pool":{combination:[4,4,1,""],d_mutant_fraction:[4,4,1,""],neighbor_effect_error:[4,4,1,""],neighbor_effect_observation:[4,4,1,""],num_of_seq:[4,4,1,""]},"k_seq.estimate":{bootstrap:[5,0,0,"-"],convergence:[5,0,0,"-"],least_squares:[5,0,0,"-"],least_squares_batch:[5,0,0,"module-0"],model_ident:[5,0,0,"-"],replicates:[5,0,0,"-"]},"k_seq.estimate.bootstrap":{Bootstrap:[5,1,1,""]},"k_seq.estimate.bootstrap.Bootstrap":{__init__:[5,2,1,""],bootstrap_num:[5,3,1,""],bs_method:[5,2,1,"id73"],bs_record_num:[5,3,1,""],bs_stats:[5,3,1,""],estimator:[5,3,1,""],grouper:[5,3,1,""],record_full:[5,3,1,""],run:[5,2,1,""]},"k_seq.estimate.convergence":{ConvergenceTester:[5,1,1,""]},"k_seq.estimate.convergence.ConvergenceTester":{__init__:[5,2,1,""],conv_init_range:[5,3,1,""],conv_reps:[5,3,1,""],conv_stats:[5,3,1,""],estimator:[5,3,1,""],run:[5,2,1,"id74"]},"k_seq.estimate.least_squares":{FitResults:[5,1,1,""],SingleFitter:[5,1,1,""]},"k_seq.estimate.least_squares.FitResults":{__init__:[5,2,1,""],convergence:[5,3,1,""],data:[5,3,1,""],estimator:[5,3,1,""],from_json:[5,2,1,""],model:[5,3,1,""],plot_fitting_curves:[5,2,1,""],plot_loss_heatmap:[5,2,1,""],point_estimation:[5,3,1,""],to_json:[5,2,1,""],to_series:[5,2,1,""],uncertainty:[5,3,1,""]},"k_seq.estimate.least_squares.SingleFitter":{__init__:[5,2,1,""],bootstrap:[5,3,1,""],bootstrap_config:[5,3,1,""],bootstrap_num:[5,3,1,""],bounds:[5,3,1,""],bs_method:[5,3,1,""],bs_record_num:[5,3,1,""],config:[5,3,1,""],convergence_test:[5,2,1,""],curve_fit_kwargs:[5,3,1,""],exclude_zero:[5,3,1,""],fit:[5,2,1,""],from_json:[5,2,1,""],init_guess:[5,3,1,""],metric:[5,3,1,""],model:[5,3,1,""],name:[5,3,1,""],opt_method:[5,3,1,""],parameter:[5,3,1,""],point_estimate:[5,2,1,""],results:[5,3,1,""],rnd_seed:[5,3,1,""],run_bootstrap:[5,2,1,""],run_replicates:[5,2,1,""],sigma:[5,3,1,""],silent:[5,3,1,""],summary:[5,2,1,""],to_dict:[5,2,1,""],to_json:[5,2,1,""],x_data:[5,3,1,""],y_data:[5,3,1,""]},"k_seq.estimate.least_squares_batch":{BatchFitResults:[5,1,1,"id21"],BatchFitter:[5,1,1,"id58"]},"k_seq.estimate.least_squares_batch.BatchFitResults":{__init__:[5,2,1,"id37"],bs_record:[5,2,1,"id38"],conv_record:[5,2,1,"id39"],data:[5,3,1,"id24"],estimator:[5,3,1,"id22"],from_json:[5,2,1,"id40"],from_pickle:[5,2,1,"id41"],generate_summary:[5,2,1,"id42"],get_FitResult:[5,2,1,"id43"],large_data:[5,3,1,"id25"],load_result:[5,2,1,"id44"],model:[5,3,1,"id23"],summary:[5,3,1,"id26"],summary_to_csv:[5,2,1,"id45"],to_json:[5,2,1,"id46"],to_pickle:[5,2,1,"id57"]},"k_seq.estimate.least_squares_batch.BatchFitter":{__init__:[5,2,1,"id67"],fit:[5,2,1,"id68"],fit_params:[5,3,1,"id66"],load_model:[5,2,1,"id69"],model:[5,3,1,"id60"],note:[5,3,1,"id64"],results:[5,3,1,"id65"],save_model:[5,2,1,"id70"],save_results:[5,2,1,"id71"],seq_to_fit:[5,3,1,"id62"],sigma:[5,3,1,"id63"],summary:[5,2,1,"id72"],x_data:[5,3,1,"id61"],y_dataframe:[5,3,1,"id59"]},"k_seq.estimate.model_ident":{ParamMap:[5,1,1,""],kendall_log:[5,4,1,""],pearson:[5,4,1,""],pearson_log:[5,4,1,""],remove_nan:[5,4,1,""],spearman:[5,4,1,""],spearman_log:[5,4,1,""]},"k_seq.estimate.model_ident.ParamMap":{fit:[5,2,1,""],get_metric_values:[5,2,1,""],load_result:[5,2,1,""],plot_map:[5,2,1,""],simulate_samples:[5,2,1,""]},"k_seq.estimate.replicates":{Replicates:[5,1,1,""]},"k_seq.estimate.replicates.Replicates":{n_replicates:[5,2,1,""],run:[5,2,1,""]},"k_seq.model":{ModelBase:[6,1,1,""],count:[6,0,0,"-"],kinetic:[6,0,0,"-"],pool:[6,0,0,"-"]},"k_seq.model.ModelBase":{func:[6,2,1,""],predict:[6,2,1,""]},"k_seq.model.count":{multinomial:[6,4,1,""]},"k_seq.model.kinetic":{BYOModel:[6,1,1,""],check_scalar:[6,4,1,""],first_order:[6,4,1,""],to_scalar:[6,4,1,""]},"k_seq.model.kinetic.BYOModel":{amount_first_order:[6,2,1,""],composition_first_order:[6,2,1,""],reacted_frac:[6,2,1,""]},"k_seq.model.pool":{PoolModel:[6,1,1,""]},"k_seq.model.pool.PoolModel":{__init__:[6,2,1,""],count_model:[6,3,1,""],count_params:[6,3,1,""],func:[6,2,1,""],kinetic_model:[6,3,1,""],kinetic_params:[6,3,1,""],note:[6,3,1,""],predict:[6,2,1,""]},"k_seq.utility":{file_tools:[7,0,0,"-"],func_tools:[7,0,0,"-"],plot_tools:[7,0,0,"-"]},"k_seq.utility.file_tools":{check_dir:[7,4,1,""],dump_json:[7,4,1,""],dump_pickle:[7,4,1,""],extract_metadata:[7,4,1,""],get_file_list:[7,4,1,""],read_json:[7,4,1,""],read_pickle:[7,4,1,""],read_table_files:[7,4,1,""],table_object_to_dataframe:[7,4,1,""]},"k_seq.utility.func_tools":{AttrScope:[7,1,1,""],FuncToMethod:[7,1,1,""],check_attr_value:[7,4,1,""],dict_flatten:[7,4,1,""],get_func_params:[7,4,1,""],is_int:[7,4,1,""],is_numeric:[7,4,1,""],is_sparse:[7,4,1,""],param_to_dict:[7,4,1,""],run_subprocess:[7,4,1,""],update_none:[7,4,1,""]},"k_seq.utility.func_tools.AttrScope":{__init__:[7,2,1,""],add:[7,2,1,""]},"k_seq.utility.plot_tools":{Presets:[7,1,1,""],ax_none:[7,4,1,""],barplot:[7,4,1,""],blue_header:[7,4,1,""],color:[7,1,1,""],format_ticks:[7,4,1,""],pairplot:[7,4,1,""],plot_curve:[7,4,1,""],plot_loss_heatmap:[7,4,1,""],regplot:[7,4,1,""],savefig:[7,4,1,""],value_to_loc:[7,4,1,""]},"k_seq.utility.plot_tools.Presets":{color_cat10:[7,2,1,""],color_pastel1:[7,2,1,""],color_tab10:[7,2,1,""],from_list:[7,2,1,""],markers:[7,2,1,""]},"k_seq.utility.plot_tools.color":{BLUE:[7,3,1,""],BOLD:[7,3,1,""],CYAN:[7,3,1,""],DARKCYAN:[7,3,1,""],END:[7,3,1,""],GREEN:[7,3,1,""],PURPLE:[7,3,1,""],RED:[7,3,1,""],UNDERLINE:[7,3,1,""],YELLOW:[7,3,1,""]},k_seq:{data:[4,0,0,"-"],estimate:[5,0,0,"-"],model:[6,0,0,"-"],utility:[7,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"001":5,"002":5,"0a_s21":0,"0a_s21_l001_r1_001":0,"0a_s21_l001_r2_001":0,"0a_s21_l002_r1_001":0,"0a_s21_l002_r2_001":0,"0a_s21_l003_r1_001":0,"0a_s21_l003_r2_001":0,"0a_s21_l004_r1_001":0,"0a_s21_l004_r2_001":0,"0a_s7":0,"0a_s7_l001_r1_001":0,"0a_s7_l001_r2_001":0,"0a_s7_l002_r1_001":0,"0a_s7_l002_r2_001":0,"0a_s7_l003_r1_001":0,"0a_s7_l003_r2_001":0,"0a_s7_l004_r1_001":0,"0a_s7_l004_r2_001":0,"100":[5,7],"1000":5,"10000":5,"10096876":0,"101":5,"10e":4,"1250":[4,7],"1250a_s16":[4,7],"1250a_s16_count":[4,7],"1260e":4,"1f77b4":4,"250e":4,"2636463":4,"2825":4,"29348173":4,"300":7,"36m":7,"479":[4,6],"50e":4,"86763":4,"91m":7,"92m":7,"93m":7,"94m":7,"95m":7,"96m":7,"981824":0,"abstract":6,"class":[0,4,5,6,7],"default":[4,5,6,7],"export":5,"float":[0,4,5,6,7],"function":[0,1,2,3,4,5,6],"import":0,"int":[0,4,5,7],"long":5,"new":4,"return":[4,5,6,7],"short":0,"static":[4,5,6,7],"true":[4,5,6,7],"try":6,"while":6,Axes:7,For:[0,1,2,4,5],Not:[4,5,7],The:[0,2,4],These:0,Use:[4,5,7],With:4,_001:0,__init__:[4,5,6,7],_count:[0,4,7],_counts_histo:0,_estim:5,_get_mask:6,aaaaaaaacaccacaca:4,aatattacatcatctatc:4,abc:6,abl:[4,5],about:[4,5],absolut:[4,6],abund:4,accept:[4,6],access:[0,5],accessor:[4,5],accord:0,accuraci:4,across:4,activ:2,actual:[4,7],adapt:0,add:[4,5,7],add_group:4,add_lin:5,add_sample_tot:4,add_spike_in:4,added:4,addit:[1,2,4],after:4,all:[0,4,5,7],allow:4,almost:5,along:4,alpha:[4,6,7],alphabet:[4,7],alreadi:5,also:0,altern:4,alwai:7,ambigu:0,aminoacyl:5,amount:[0,4,6],amount_first_ord:[4,6],anaconda:2,analys:4,analysi:[1,4,5],analyz:[0,1,4],ani:[1,4,6,7],api:0,appli:[4,5,6],applic:[4,7],arg:[4,6,7],argument:[4,5,6,7],as_dict:4,aspect:6,assai:1,assign:7,associ:5,assumpt:5,asymptot:6,attr:7,attr_dict:7,attr_kwarg:7,attribut:[4,5,7],attrscop:[4,5,7],automat:6,avail:[4,6],awai:4,ax_non:7,axi:[4,7],back:[4,7],bar:4,barplot:[4,7],barplot_kwarg:[4,7],base:[4,5,6,7],base_t:4,batch:5,batchfitresult:5,batchfitt:5,befor:[0,2,4],best:7,between:[6,7],bin:4,black_list:[4,7],blue:7,blue_head:7,bold:7,bool:[4,5,6,7],bootstrap:[1,7],bootstrap_config:5,bootstrap_num:5,both:0,bound:5,brace:0,broadcast:[6,7],bs_ci95:4,bs_method:5,bs_param1_2:5,bs_param1_mean:5,bs_param1_std:5,bs_record:5,bs_record_num:5,bs_stat:5,built:4,byo:[4,6],byomodel:[4,6],c95:4,c_i:6,calcul:[4,5,6,7],call:4,callabl:[4,5,6,7],can:[0,2,4,5,6,7],caus:[4,7],cax_po:5,cell:0,center:4,chang:[2,4,6],character:0,check:[1,6,7],check_attr_valu:7,check_dir:7,check_scalar:6,choos:5,citat:4,classmethod:[4,5],cmd:7,code:4,coeffici:6,col:4,col_nam:7,collect:[4,5,6,7],color:[4,7],color_cat10:7,color_map:4,color_pastel1:7,color_tab10:7,colorbar:[5,7],column:[4,5],com:4,combin:4,common:7,commonli:[4,6],compare_t:4,compat:5,complet:[0,1],compo_lognorm:4,compon:4,composit:[4,6],composition_first_ord:6,compress:5,concentr:[4,5,6,7],cond:0,conda:[],condit:4,conduct:5,confid:5,config:5,configur:5,consid:4,const_err:5,constitu:6,consumpt:5,contain:[0,1,3,4,5,6,7],control:[0,4],conv_init_rang:5,conv_record:5,conv_rep:[4,5],conv_stat:5,convent:[4,7],converg:7,convergence_test:5,convergencetest:5,convers:6,convert:[4,5,6,7],core:[4,5],correlationw:7,correspond:[5,7],cost:7,cost_fn:7,could:[0,4,5],count:[1,2,4],count_fil:[0,4],count_model:[4,6],count_param:6,countfil:4,covari:5,cpu:4,crate:2,creat:[2,4,5,7],created_tim:4,criteria:0,cross_table_compar:4,csv:[2,4,5,7],ctacgaattc:0,ctgcagtgaa:0,current:[5,6],curv:[4,5,7],curve_fit:5,curve_fit_kwarg:5,curve_kwarg:7,curve_label:7,custom:5,cyan:7,d_mutant_fract:4,darkcyan:7,data:[1,2,3,5,6,7],data_not:4,data_unit:4,datafram:[4,5,7],datapoint_color:7,datapoint_kwarg:7,datapoint_label:7,dataset:[0,4,5],dataset_dir:4,dataset_metadata:4,datetim:4,debug:5,dedupl:[0,2,4,5],defin:[0,4],degrad:[4,6],delet:4,densiti:4,depend:[2,4,5],deprec:4,depth:4,describ:4,design:[0,4],detect:4,deviat:4,dict:[4,5,7],dict_flatten:7,dictionari:[4,5,7],differ:[0,4,5],digit:7,dim:7,dimens:[4,6,7],direct:0,directli:[2,5,7],directori:[2,4,5,7],discard:[0,4],disk:5,dist_typ:4,distanc:4,distgener:4,distribut:[4,6],dna:[0,4],document:[0,1],doe:[4,5],domain:[4,7],domain_nam:[4,7],dope:4,download:2,dpi:7,draw:[4,5,6],drive:5,dry_run:4,due:[4,5],dump_json:7,dump_pickl:7,duplic:4,dure:[0,4,5],e45756:7,each:[0,4,5,6,7],easydiv:[2,4],edit:4,edu:1,effect:4,either:5,elementwis:6,empir:5,empti:4,end:[4,7],energi:7,enforc:4,entri:4,environ:[0,1,2],equal:4,equation_loc:7,error:[0,4,7],estim:[1,3,4,6,7],estimatorbas:5,eta:4,etc:6,even:4,evenli:4,exampl:[0,4,7],except:[5,7],exclud:[4,5,7],exclude_zero:5,exist:[5,6,7],exlud:4,exp:[0,4,6],exp_cond:0,exp_rep:[0,4,7],expect:[0,4],experi:[0,4,6],experiment:4,extend:[1,4],extern:0,extra:[4,5],extract:[0,4,7],extract_metadata:[4,7],factor:[4,6],fail:0,fals:[4,5,6,7],fasta:0,fastq:[0,1,2,4],fastq_root:4,fastq_to_count:4,fig_save_to:4,figsiz:[4,5,7],figur:[4,7],file:[2,4,5,7],file_list:[4,7],file_path:[4,5,7],file_root:7,filter_axi:4,finish:5,finite_onli:5,first:[0,4,6,7],first_ord:6,fit:[1,3,4,7],fit_param:5,fit_summari:4,fitresult:5,fitting_kwarg:5,fitting_r:4,fix:[4,6,7],fixed_param:7,flatten:5,flow:0,folder:[0,4,5,7],foler:4,follow:[0,3,4,6],fontsiz:[4,5,7],form:7,format:[0,4,5,7],format_tick:7,forward:[0,4],forward_prim:4,found:5,fraction:[4,5,6],frame:4,frequent:[0,3],from:[0,1,2,4,5,6,7],from_count_fil:[0,4],from_json:[4,5],from_list:7,from_pickl:[4,5],full:[4,5,6,7],full_path:7,full_tabl:4,func:[6,7],func_tool:4,functomethod:[4,7],further:[0,5],gaussian:4,gener:[4,5,6,7],generate_summari:5,get:[1,4,5,6,7],get_est_result:4,get_file_list:7,get_fitresult:5,get_fold_rang:4,get_func_param:7,get_group:4,get_metric_valu:5,get_pct_gaussian_error:4,get_tabl:4,get_uncertainty_accuraci:4,gggggggaacgcatttcacgg:0,gggggggaagactccggaacg:0,gggggggacgttcaccggcaa:0,gggggggagtaggactgcaaa:0,ggggggggattcatgactatt:0,git:2,github:4,give:[4,7],given:[4,5,6,7],green:7,grid:5,gridsiz:5,group:[4,5,7],group_memb:4,group_nam:4,group_title_po:4,grouper:5,groupercollect:4,guess:5,gzip:5,ham:4,handl:4,hard:5,has:2,hash:5,have:[0,4,5,6,7],header:7,heatmap:[5,7],heavili:[0,4],help:0,here:0,heterogen:4,high:[4,7],higher:5,hist_kwarg:4,histo:0,histogram:[0,4],http:[2,4],ichen:4,identifi:[0,5],iii:7,illumina:0,implement:[0,6,7],includ:[0,4,5,6,7],indent:7,independ:4,index:[1,4,5],indic:[0,4,5,7],individu:4,infer:5,info:[4,5,7],infor:4,inform:[4,5],init:5,init_guess:5,initi:[4,5,6,7],innest:7,inplac:4,input:[0,4,5,6],input_count:4,input_sample_nam:[0,4],insert:4,instal:[0,1],instanc:[4,5],instead:4,int_tick_onli:7,integr:0,interv:5,introduc:[0,4],investig:4,is_int:7,is_numer:7,is_spars:7,issu:1,job:5,join:[2,4],join_first:4,joint:4,json:[4,5,7],json_path:5,jsonfi:5,k_95:4,k_seq:[0,1,3],kei:[4,7],kendall_log:5,key_list:7,keyword:[4,5,7],kinet:[3,4,5,7],kinetic_model:[4,6],kinetic_param:6,kwarg:[4,5,7],kwargs_lin:7,kwargs_scatt:7,lab:4,label:[0,4,7],label_map:4,label_mapp:4,landscap:7,lane:[0,4],larg:5,large_data:5,large_dataset:5,larger:5,least:[1,3],left:[5,7],legend:[5,7],legend_loc:[5,7],len:6,length:[0,4,5,6,7],less:4,letter_book_s:4,level:[5,7],like:[4,5,6,7],limit:7,line_label:5,link:5,list:[4,5,6,7],load:[4,5],load_model:5,load_result:5,load_seqtable_from_count_fil:4,loc:4,local:5,locat:7,log:[0,4,5,7],logi:4,lognorm:4,logx:4,loss:5,low:[4,7],lower:[5,7],lpha:6,main:0,maintain:6,major:[4,7],major_curve_kwarg:7,major_curve_label:7,major_param:7,manipul:[0,4],manual:7,map:5,marker:7,marker_s:4,match:[0,4,7],matrix:[4,5,6],max:5,max_len:0,maxim:7,mean:[4,5,7],mean_count:4,measur:4,member:7,memori:5,metadata:[0,4,7],method:[0,4,5,7],metric:5,metric_label:5,might:6,migrat:6,min:[4,5,6],min_len:0,miniconda:2,minim:2,miscellan:[1,3],miss:[4,5,7],modalbas:6,model:[1,3,4,5,7],model_config:5,model_kwarg:5,model_path:5,modelbas:6,modul:[1,3,5],molecul:[0,4],more:4,most:2,multinomi:[4,6],multipl:[4,6,7],mutant:4,mutation_r:4,n_core:5,n_input:4,n_replic:5,n_row:4,name:[0,4,5,6,7],name_templ:[0,4],ncol:4,ndarrai:[4,6],need:[4,5,6],neg:[4,5,6],neighbor:4,neighbor_effect_error:4,neighbor_effect_observ:4,next:0,noambiguityfilt:0,none:[4,5,6,7],normal:[4,6],note:[0,4,5,6,7],notic:[4,5,6,7],nrow:4,nucleotid:0,num:7,num_of_seq:4,number:[0,4,5,6],number_onli:4,numer:[4,7],obj:[5,7],object:[4,5,6,7],observ:4,onc:2,one:[4,7],ones:6,onli:[1,4,5,7],opt_method:5,optim:5,option:[4,5,7],order:[4,6,7],org:2,organ:0,origin:[0,4],other:[4,5,7],otherwis:[4,7],out:7,outer:6,output:[0,4,5,6,7],output_dir:5,output_path:4,over:[4,6,7],overal:5,overlap:[0,4],overview:4,overwrit:[5,6],p0_gener:4,p0_loc:4,p0_scale:4,packag:[0,2,4],packg:2,pair:4,pairplot:7,pairwis:[4,7],panda:4,pandas_abs_match:4,pandaseq:[0,2,4],paper:4,parallel:[0,5],parallel_cor:5,param1:[5,7],param1_log:5,param1_nam:5,param1_rang:[5,7],param2:[5,7],param2_log:5,param2_nam:5,param2_rang:[5,7],param:[4,5,6,7],param_gener:4,param_log:7,param_nam:[5,7],param_sample_from_df:4,param_t:[4,6],param_to_dict:7,paramet:[4,5,6,7],parammap:5,parent_kei:7,pars:[0,4,5,7],pass:[5,7],path:[0,4,5,7],path_to_fold:5,path_to_pickl:5,pattern:[0,4,7],pattern_filt:[0,4],pcov:5,pct_re:5,peak:4,pearson:5,pearson_log:5,per:4,percent:5,percentil:4,perform:[0,5],perturb:5,pick:[5,7],pickl:[4,5,7],pip:2,pipelin:[0,4],pkl:5,pleas:[0,1,2],plot:[4,5,7],plot_curv:7,plot_dist:4,plot_fitting_curv:5,plot_kwarg:5,plot_loss_heatmap:[5,7],plot_map:5,plot_on:5,plot_spike_in_frac:4,plot_total_count:4,plot_unique_seq:4,plt:7,plugin:4,point:[4,5,6,7],point_est:4,point_est_csv:4,point_estim:5,pool:[0,1,3,4],poolmodel:6,poolparamgener:4,posit:[0,7],possibl:[5,6],pre:4,pre_process:4,pred:4,pred_typ:4,predict:6,prefer:5,preprocess:[1,2,3],preserv:4,preset:7,previou:4,primer:4,print:4,probabl:6,problem:6,procedur:4,process:[0,1,2,4,5,6],product:6,progress:[0,3],project:[2,7],prop_list:7,properti:[4,5],proport:[4,5],protein:0,provid:[4,5],proxi:5,purpl:7,pypi:2,python:[0,1],qpcr:4,qualiti:[0,4],quantif:[1,4],quantifi:[1,4],question:1,quick:[0,1,4],r4a:0,r4b:[0,4,7],radiu:4,rais:[4,7],random:[4,5,6],randomli:5,rang:[4,5,7],rate:4,ratio:4,raw:4,react:[4,5,6],reacted_frac:6,reaction:[1,4,6],read:[1,2,4,5,7],read_count_fil:4,read_json:7,read_pickl:7,read_table_fil:7,readabl:[4,5],readi:2,reassign:5,recommend:[2,5],record:5,record_ful:5,recov:5,red:7,refer:0,region:[0,4,7],regplot:7,rel:4,rel_err:5,rel_r:5,remov:[0,4],remove_empti:4,remove_nan:5,remove_zero:4,rep:[0,4],rep_result:5,rep_variance_scatt:4,repeat:[4,5],replac:4,replic:[4,7],repo:2,report:[1,5],repres:5,reproduc:5,requir:[2,4,5,6],required_onli:7,resampl:[4,5],reserv:4,residu:5,resolut:[5,7],result:[4,5,7],result_dir:4,result_folder_path:5,result_path:5,retriev:5,revers:[0,4],reverse_prim:4,ribozym:5,rich:4,right:7,rna:0,rnd_seed:5,root:[4,5,7],row:[4,5,7],run:[0,1,4,5],run_bootstrap:5,run_repl:5,run_subprocess:7,same:[4,5,6,7],sampl:[4,5,6,7],sample_entropy_scatterplot:4,sample_from_datafram:4,sample_from_iid_dist:4,sample_id:0,sample_info:4,sample_list:4,sample_metadata:4,sample_n:5,sample_nam:[4,7],sample_num:6,sample_overview:4,sample_overview_plot:4,sample_rel_abun_hist:4,sample_spike_in_ratio_scatterplot:4,sample_tot:4,sample_total_reads_barplot:4,sample_unique_seqs_barplot:4,samplefilt:0,save:[0,4,5,7],save_fig_to:[4,7],save_model:5,save_result:5,save_to:[4,5],save_to_fil:5,savefig:7,scalar:6,scale:[4,5,7],scan:7,scan_rang:7,scatter:[4,5,7],scatter_kwarg:4,scipi:5,scope:[5,7],script:0,seaborn:7,search:[4,7],secur:5,seed:[4,5,6],select:[0,5],self:[5,7],sep:7,sep_plot:4,separ:[2,4,5,7],seq1:5,seq2:5,seq:[0,2,4,5,6],seq_length_dist:4,seq_list:4,seq_mean_value_detected_samples_scatterplot:4,seq_metadata:4,seq_overview:4,seq_rep:[4,7],seq_tabl:[4,7],seq_to_fit:[4,5],seq_to_hash:5,seq_vari:4,seqdata:[0,4,7],seqdataanalyz:4,seqlengthfilt:0,seqtabl:4,seqtable_path:4,seqtableanalyz:4,sequenc:[2,4,5,6],sequence_count:4,sequenceseq:4,sequencing_depth:4,seri:[4,5,7],set:[0,1,4,5,6,7],setup:4,sever:5,shape:6,sheet:7,shen:1,should:[0,4,5,6,7],show:[4,7],sigma:5,signatur:6,silent:5,simga:5,similar:4,simplifi:4,simul:[4,5],simulate_count:4,simulate_on_byo_doped_condition_from_exp_result:4,simulate_sampl:5,simulate_w_byo_doped_condition_from_param_dist:4,simulationresult:4,singl:[4,5,6,7],singlefitt:5,size:[4,5],skip:5,slice:4,slice_t:4,small:5,smaller:5,smpl:0,some:[0,4,6,7],sort:4,sort_bi:[0,4],sourc:4,space:[5,7],spars:4,spearman:5,spearman_log:5,specif:0,specifi:[4,7],speed:5,spike:[0,4],spike_in:4,spike_in_amount:4,spike_in_seq:4,spikeinnorm:4,split:4,spread:4,squar:[1,3,7],squeez:6,stabl:4,standard:[0,4],start:1,stat:5,statist:[0,4],storag:5,store:[0,4,5,6],str:[4,5,6,7],straight:4,strategi:5,stratifi:5,stream:5,stream_to:5,string:[0,4,5,7],structur:[4,5,6],sub:4,subclass:6,submodul:[4,5],subprocess:4,subsampl:[4,5,7],substr:0,substrat:6,subtabl:4,successfulli:5,suffix:5,suggest:5,suitabl:[4,5],sum:[4,6],summar:[4,5],summari:[4,5],summary_to_csv:5,suppli:5,support:[4,5,6],survei:4,tabl:[0,2,4,5],table_nam:[4,7],table_object_to_datafram:7,take:[4,5,7],tar:5,tar_file_nam:5,tarbal:5,tarfil:5,target:4,templat:[4,7],tensor:6,test:5,them:[4,7],thi:[0,3,4,5,7],thread:[0,4,5],three:5,through:[2,5],tick:7,tick_formatt:7,tick_num:7,time:[4,5,6],timestamp:4,to_dict:5,to_json:[4,5],to_pickl:[4,5],to_scalar:6,to_seri:5,todo:[4,6],togeth:0,too:5,tool:[0,4],top:5,total:[0,4,6],total_amount:4,total_amount_error:4,total_count:4,total_dna_error_r:4,total_read:4,totalamountnorm:4,trf:5,trigger:5,trim:[0,2,4],tripl:4,triplic:4,truth:4,tsv:7,tupl:[5,7],tutori:[1,3],two:[0,4,5,7],txt:[0,4,7],type:[0,4,5,6,7],ucsb:[1,4],uncertainti:[1,4,5],under:[0,3,4,5,7],underlin:7,undesir:0,uneven:4,uniform:[4,5],uniq_seq_num:[4,6],uniqu:[0,4],unique_seq:4,unit:4,unreact:4,updat:[0,3,4,7],update_analysi:4,update_bi:7,update_non:7,upper:[5,7],usag:0,use:[0,2,4,5,7],use_spars:4,used:[0,4,5,6],uses:4,using:[1,4,5],util:[1,3,4],valid:[4,7],valu:[4,5,6,7],value_to_loc:7,variabl:[4,6,7],varianc:[4,5],variant:4,vars_lim:7,vars_log:7,vars_nam:7,vector:[4,6],verbos:5,veri:5,view:4,visual:[4,7],warn:5,weight:[4,5],welcom:1,well:5,were:[4,6],when:5,where:[0,4,6],which:[4,5,6],wildcard:4,with_lin:7,within:5,without:4,work:[0,3,5],wrapper:[6,7],write:4,x1b:7,x_data:5,x_label:[4,5,7],x_lim:[5,7],x_log:4,x_tick_formatt:7,x_unit:[0,4],x_valu:[0,4,5],xaxi:4,xlabel:[4,7],xlim:4,xlog:[4,7],xls:7,xlsx:7,y_data:5,y_datafram:5,y_enforce_posit:5,y_label:[4,5,7],y_lim:[5,7],y_log:4,y_tick_formatt:7,y_valu:5,yellow:7,yet:4,ylabel:[4,7],ylim:4,ylog:[4,7],yml:2,your:0,yticklabel:7,yune:1,yuningshen:1,z_log:7,zero:[4,5]},titles:["Getting Started Tutorial","<code class=\"docutils literal notranslate\"><span class=\"pre\">k-seq</span></code>: a package for kinetic measurements with high throughput sequencing","Installation","<code class=\"docutils literal notranslate\"><span class=\"pre\">k-seq</span></code> package","<code class=\"docutils literal notranslate\"><span class=\"pre\">k_seq.data</span></code>: data preprocessing","<code class=\"docutils literal notranslate\"><span class=\"pre\">k_seq.estimate</span></code>: least squares fitting","<code class=\"docutils literal notranslate\"><span class=\"pre\">k_seq.model</span></code>: kinetics and pool models","<code class=\"docutils literal notranslate\"><span class=\"pre\">k_seq.utility</span></code>: miscellaneous utility functions"],titleterms:{"function":7,"new":[],addit:0,analysi:0,api:1,bootstrap:[0,5],calcul:0,complet:2,conda:2,contact:1,content:1,converg:5,count:[0,6],creat:[],data:[0,4],depend:0,easydiv:0,end:0,environ:[],estim:5,file:0,file_tool:7,filter:[0,4],fit:[0,5],func_tool:7,get:0,grouper:4,high:1,instal:2,join:0,k_seq:[4,5,6,7],kinet:[1,6],least:5,least_squar:5,least_squares_batch:5,load:0,manual:2,measur:1,miscellan:7,model:[0,6],model_id:5,modul:[4,6,7],onli:2,option:2,packag:[1,3],pair:0,pip:[],plot_tool:7,pool:6,preprocess:[0,4],python:2,quantif:0,quantifi:0,read:0,refer:1,replic:5,sampl:0,seq:[1,3],seq_data:4,seq_data_analyz:4,sequenc:[0,1],simu:4,squar:5,start:0,tabl:1,through:[],throughput:1,tutori:0,using:0,util:7,variabl:0,variant_pool:4,visual:5,visualz:4}})