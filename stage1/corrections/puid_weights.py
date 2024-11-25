import numpy as np
import pandas as pd

# def puid_weights(evaluator, year, jets, pt_name, jet_puid_opt, jet_puid, numevents):
#     yearname = year
#     if "2017corrected" in jet_puid_opt:
#         # h_eff_name_L = f"h2_eff_mc{year}_L"
#         # h_sf_name_L = f"h2_eff_sf{year}_L"
#         # h_eff_name_T = f"h2_eff_mc{year}_T"
#         # h_sf_name_T = f"h2_eff_sf{year}_T"
#         # jetpt = jets[pt_name].values
#         # jeteta = jets.eta.values
#         # puid_eff_L = evaluator[h_eff_name_L](jetpt, jeteta)
#         # puid_sf_L = evaluator[h_sf_name_L](jetpt, jeteta)
#         # puid_eff_T = evaluator[h_eff_name_T](jetpt, jeteta)
#         # puid_sf_T = evaluator[h_sf_name_T](jetpt, jeteta)
#         # # puid_eff_L = evaluator[h_eff_name_L](jets[pt_name], jets.eta)
#         # # puid_sf_L = evaluator[h_sf_name_L](jets[pt_name], jets.eta)
#         # # puid_eff_T = evaluator[h_eff_name_T](jets[pt_name], jets.eta)
#         # # puid_sf_T = evaluator[h_sf_name_T](jets[pt_name], jets.eta)

#         # jets_passed_L = (
#         #     (jets[pt_name] > 25)
#         #     & (jets[pt_name] < 50)
#         #     & jet_puid
#         #     & ((abs(jets.eta) < 2.6) | (abs(jets.eta) > 3.0))
#         # )
#         # jets_failed_L = (
#         #     (jets[pt_name] > 25)
#         #     & (jets[pt_name] < 50)
#         #     & (~jet_puid)
#         #     & ((abs(jets.eta) < 2.6) | (abs(jets.eta) > 3.0))
#         # )
#         # jets_passed_T = (
#         #     (jets[pt_name] > 25)
#         #     & (jets[pt_name] < 50)
#         #     & jet_puid
#         #     & ((abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0))
#         # )
#         # jets_failed_T = (
#         #     (jets[pt_name] > 25)
#         #     & (jets[pt_name] < 50)
#         #     & (~jet_puid)
#         #     & ((abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0))
#         # )

#         # pMC_L = (
#         #     puid_eff_L[jets_passed_L].prod() * (1.0 - puid_eff_L[jets_failed_L]).prod()
#         # )
#         # pMC_T = (
#         #     puid_eff_T[jets_passed_T].prod() * (1.0 - puid_eff_T[jets_failed_T]).prod()
#         # )

#         # pData_L = (
#         #     puid_eff_L[jets_passed_L].prod()
#         #     * puid_sf_L[jets_passed_L].prod()
#         #     * (1.0 - puid_eff_L[jets_failed_L] * puid_sf_L[jets_failed_L]).prod()
#         # )
#         # pData_T = (
#         #     puid_eff_T[jets_passed_T].prod()
#         #     * puid_sf_T[jets_passed_T].prod()
#         #     * (1.0 - puid_eff_T[jets_failed_T] * puid_sf_T[jets_failed_T]).prod()
#         # )

#         # puid_weight = np.ones(numevents)
#         # puid_weight[pMC_L * pMC_T != 0] = np.divide(
#         #     (pData_L * pData_T)[pMC_L * pMC_T != 0], (pMC_L * pMC_T)[pMC_L * pMC_T != 0]
#         # )
#         h_eff_name_L = f"h2_eff_mc{year}_L"
#         h_sf_name_L = f"h2_eff_sf{year}_L"
#         h_eff_name_T = f"h2_eff_mc{year}_T"
#         h_sf_name_T = f"h2_eff_sf{year}_T"
#         jetpt = jets[pt_name].values
#         jeteta = jets.eta.values
#         puid_eff_L = evaluator[h_eff_name_L](jetpt, jeteta)
#         puid_sf_L = evaluator[h_sf_name_L](jetpt, jeteta)
#         puid_eff_T = evaluator[h_eff_name_T](jetpt, jeteta)
#         puid_sf_T = evaluator[h_sf_name_T](jetpt, jeteta)

#         jets_passed_L = (
#             (jets[pt_name] > 25)
#             & (jets[pt_name] < 50) # For 2017 the loos PUID is applied to all jets
#             & jet_puid
#             & ((abs(jets.eta) < 2.6) | (abs(jets.eta) > 3.0))
#         )
#         jets_failed_L = (
#             (jets[pt_name] > 25)
#             & (jets[pt_name] < 50)
#             & (~jet_puid)
#             & ((abs(jets.eta) < 2.6) | (abs(jets.eta) > 3.0))
#         )
#         jets_passed_T = (
#             (jets[pt_name] > 25)
#             #& (jets[pt_name] < 50)
#             & jet_puid
#             & ((abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0))
#         )
#         jets_failed_T = (
#             (jets[pt_name] > 25)
#             #& (jets[pt_name] < 50)
#             & (~jet_puid)
#             & ((abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0))
#         )
#         # testing possible improvement start -------------------------------------
#         # obtain jet puid weights for Loose wp
#         jets["puid_eff_L"] = puid_eff_L
#         jets["oneminuspuid_eff_L"] = 1.0-puid_eff_L
#         jets["puid_sf_L"] = puid_sf_L
#         jets["eff_sf_L"] = puid_sf_L * puid_eff_L
#         jets["oneminus_eff_sf_L"] = 1.0-(puid_sf_L * puid_eff_L)
#         pMC_passed_L = pd.Series(1, index=range(0, numevents))
#         pMC_failed_L = pd.Series(1, index=range(0, numevents))
#         pMC_passed_bare_L = jets[jets_passed_L==True].groupby('entry').puid_eff_L.prod()
#         pMC_failed_bare_L =(jets.loc[jets_failed_L==True].groupby('entry').oneminuspuid_eff_L).prod()
#         pMC_passed_L = pd.Series(1, index=range(0, numevents))
#         pMC_failed_L = pd.Series(1, index=range(0, numevents))
#         pMC_passed_L.update(pMC_passed_bare_L)
#         pMC_failed_L.update(pMC_failed_bare_L)
#         pSF_L = pd.Series(1, index=range(0, numevents))
#         pfailSF_L = pd.Series(1, index=range(0, numevents))
#         pSF_bare_L = jets[jets_passed_L==True].groupby('entry').puid_sf_L.prod()
#         pfailSF_bare_L = (jets[jets_failed_L==True].groupby('entry').oneminus_eff_sf_L).prod()
#         pSF_L.update(pSF_bare_L)
#         pfailSF_L.update(pfailSF_bare_L)


#         pMC_L = pMC_passed_L * pMC_failed_L
#         pData_L = (
#             pMC_passed_L
#             * pSF_L
#             * pfailSF_L
#         )
        
#         # obtain jet puid weights for Tight wp
#         jets["puid_eff_T"] = puid_eff_T
#         jets["oneminuspuid_eff_T"] = 1.0-puid_eff_T
#         jets["puid_sf_T"] = puid_sf_T
#         jets["eff_sf_T"] = puid_sf_T * puid_eff_T
#         jets["oneminus_eff_sf_T"] = 1.0-(puid_sf_T * puid_eff_T)
#         pMC_passed_T = pd.Series(1, index=range(0, numevents))
#         pMC_failed_T = pd.Series(1, index=range(0, numevents))
#         pMC_passed_bare_T = jets[jets_passed_T==True].groupby('entry').puid_eff_T.prod()
#         pMC_failed_bare_T =(jets.loc[jets_failed_T==True].groupby('entry').oneminuspuid_eff_T).prod()
#         pMC_passed_T = pd.Series(1, index=range(0, numevents))
#         pMC_failed_T = pd.Series(1, index=range(0, numevents))
#         pMC_passed_T.update(pMC_passed_bare_T)
#         pMC_failed_T.update(pMC_failed_bare_T)
#         pSF_T = pd.Series(1, index=range(0, numevents))
#         pfailSF_T = pd.Series(1, index=range(0, numevents))
#         pSF_bare_T = jets[jets_passed_T==True].groupby('entry').puid_sf_T.prod()
#         pfailSF_bare_T = (jets[jets_failed_T==True].groupby('entry').oneminus_eff_sf_T).prod()
#         pSF_T.update(pSF_bare_T)
#         pfailSF_T.update(pfailSF_bare_T)


#         pMC_T = pMC_passed_T * pMC_failed_T
#         pData_T = (
#             pMC_passed_T
#             * pSF_T
#             * pfailSF_T
#         )
        
#         # merge jetpuid for loose and tight
#         pMC = pMC_L * pMC_T
#         pData = pData_L * pData_T 
        
#         #print(pMC_passed)
#         #print(pData)
#         puid_weight = np.ones(numevents)
#         puid_weight[pMC != 0] = np.divide(pData[pMC != 0], pMC[pMC != 0])
#     else:
#         # wp_dict = {"loose": "L", "medium": "M", "tight": "T"}
#         # wp = wp_dict[jet_puid_opt]
#         # h_eff_name = f"h2_eff_mc{year}_{wp}"
#         # h_sf_name = f"h2_eff_sf{year}_{wp}"
#         # jetpt = jets[pt_name].values
#         # jeteta = jets.eta.values
#         # # puid_eff = evaluator[h_eff_name](jets[pt_name], jets.eta)
#         # # puid_sf = evaluator[h_sf_name](jets[pt_name], jets.eta)
#         # puid_eff = evaluator[h_eff_name](jetpt, jeteta)
#         # puid_sf = evaluator[h_sf_name](jetpt, jeteta)
#         # # jets_passed = (jets[pt_name] > 25) & (jets[pt_name] < 50) & jet_puid
#         # # jets_failed = (jets[pt_name] > 25) & (jets[pt_name] < 50) & (~jet_puid)
#         # jets_passed = (jetpt > 25) & (jetpt < 50) & jet_puid
#         # jets_failed = (jetpt > 25) & (jetpt < 50) & (~jet_puid)

#         # pMC = puid_eff[jets_passed].prod() * (1.0 - puid_eff[jets_failed]).prod()
#         # print(f"pMC: {pMC}")
#         # pData = (
#         #     puid_eff[jets_passed].prod()
#         #     * puid_sf[jets_passed].prod()
#         #     * (1.0 - puid_eff[jets_failed] * puid_sf[jets_failed]).prod()
#         # )
#         # puid_weight = np.ones(numevents)
#         # puid_weight[pMC != 0] = np.divide(pData[pMC != 0], pMC[pMC != 0])


#         wp_dict = {"loose": "L", "medium": "M", "tight": "T"}
#         wp = wp_dict[jet_puid_opt]
#         h_eff_name = f"h2_eff_mc{yearname}_L"
#         h_sf_name = f"h2_eff_sf{yearname}_L"
#         jetpt = jets[pt_name].values
#         jeteta = jets.eta.values
#         puid_eff = evaluator[h_eff_name](jetpt, jeteta)
#         puid_sf = evaluator[h_sf_name](jetpt, jeteta)
#         jets["puid_eff"] = puid_eff
#         jets["oneminuspuid_eff"] = 1.0-puid_eff
#         jets["puid_sf"] = puid_sf
#         jets["eff_sf"] = puid_sf * puid_eff
#         jets["oneminus_eff_sf"] = 1.0-(puid_sf * puid_eff)
#         #with open(f'puideff.txt', 'w') as f:
#             #print(jets["puid_eff"], file=f)
#         jets_passed = (jets[pt_name] > 25) & (jets[pt_name] < 50) & jet_puid
#         jets_failed = (jets[pt_name] > 25) & (jets[pt_name] < 50) & (~jet_puid)
#         #with open(f'jetspass.txt', 'w') as f:
#             #print(jets_passed, file=f) 
#         #with open(f'jetsfail.txt', 'w') as f:
#             #print(jets_failed, file=f) 
#         pMC_failed = np.ones(numevents)
#         pMC_passed_bare = jets[jets_passed==True].groupby('entry').puid_eff.prod()
#         pMC_failed_bare =(jets.loc[jets_failed==True].groupby('entry').oneminuspuid_eff).prod()
#         pMC_passed = pd.Series(1, index=range(0, numevents))
#         pMC_failed = pd.Series(1, index=range(0, numevents))
#         pMC_passed.update(pMC_passed_bare)
#         pMC_failed.update(pMC_failed_bare)
#         pSF = pd.Series(1, index=range(0, numevents))
#         pfailSF = pd.Series(1, index=range(0, numevents))
#         pSF_bare = jets[jets_passed==True].groupby('entry').puid_sf.prod()
#         pfailSF_bare = (jets[jets_failed==True].groupby('entry').oneminus_eff_sf).prod()
#         pSF.update(pSF_bare)
#         pfailSF.update(pfailSF_bare)
        
#         #print(pMC_failed)
#         pMC = pMC_passed * pMC_failed

#         pData = (
#             pMC_passed
#             * pSF
#             * pfailSF
#         )
#         #print(pMC_passed)
#         #print(pData)
#         puid_weight = np.ones(numevents)
#         puid_weight[pMC != 0] = np.divide(pData[pMC != 0], pMC[pMC != 0])
        
#     return puid_weight





def puid_weights(evaluator, year, jets, pt_name, jet_puid_opt, jet_puid, numevents):
    yearname = year
    if "2017corrected" in jet_puid_opt:
        print("doing the 2017corrected jetPUID method !")
        h_eff_name_L = f"h2_eff_mcUL{year}_L"
        h_sf_name_L = f"h2_eff_sfUL{year}_L"
        h_eff_name_T = f"h2_eff_mcUL{year}_T"
        h_sf_name_T = f"h2_eff_sfUL{year}_T"
        jetpt = jets[pt_name].values
        jeteta = jets.eta.values
        puid_eff_L = evaluator[h_eff_name_L](jetpt, jeteta)
        puid_sf_L = evaluator[h_sf_name_L](jetpt, jeteta)
        puid_eff_T = evaluator[h_eff_name_T](jetpt, jeteta)
        puid_sf_T = evaluator[h_sf_name_T](jetpt, jeteta)

        jets_passed_L = (
            (jets[pt_name] > 25)
            & (jets[pt_name] < 50) # For 2017 the loos PUID is applied to all jets
            & jet_puid
            & ((abs(jets.eta) < 2.6) | (abs(jets.eta) > 3.0))
        )
        jets_failed_L = (
            (jets[pt_name] > 25)
            & (jets[pt_name] < 50)
            & (~jet_puid)
            & ((abs(jets.eta) < 2.6) | (abs(jets.eta) > 3.0))
        )
        jets_passed_T = (
            (jets[pt_name] > 25)
            #& (jets[pt_name] < 50)
            & jet_puid
            & ((abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0))
        )
        jets_failed_T = (
            (jets[pt_name] > 25)
            #& (jets[pt_name] < 50)
            & (~jet_puid)
            & ((abs(jets.eta) > 2.6) & (abs(jets.eta) < 3.0))
        )
        # testing possible improvement start -------------------------------------
        # obtain jet puid weights for Loose wp
        jets["puid_eff_L"] = puid_eff_L
        jets["oneminuspuid_eff_L"] = 1.0-puid_eff_L
        jets["puid_sf_L"] = puid_sf_L
        jets["eff_sf_L"] = puid_sf_L * puid_eff_L
        jets["oneminus_eff_sf_L"] = 1.0-(puid_sf_L * puid_eff_L)
        pMC_passed_L = pd.Series(1, index=range(0, numevents))
        pMC_failed_L = pd.Series(1, index=range(0, numevents))
        pMC_passed_bare_L = jets[jets_passed_L==True].groupby('entry').puid_eff_L.prod()
        pMC_failed_bare_L =(jets.loc[jets_failed_L==True].groupby('entry').oneminuspuid_eff_L).prod()
        pMC_passed_L = pd.Series(1, index=range(0, numevents))
        pMC_failed_L = pd.Series(1, index=range(0, numevents))
        pMC_passed_L.update(pMC_passed_bare_L)
        pMC_failed_L.update(pMC_failed_bare_L)
        pSF_L = pd.Series(1, index=range(0, numevents))
        pfailSF_L = pd.Series(1, index=range(0, numevents))
        pSF_bare_L = jets[jets_passed_L==True].groupby('entry').puid_sf_L.prod()
        pfailSF_bare_L = (jets[jets_failed_L==True].groupby('entry').oneminus_eff_sf_L).prod()
        pSF_L.update(pSF_bare_L)
        pfailSF_L.update(pfailSF_bare_L)


        pMC_L = pMC_passed_L * pMC_failed_L
        pData_L = (
            pMC_passed_L
            * pSF_L
            * pfailSF_L
        )
        
        # obtain jet puid weights for Tight wp
        jets["puid_eff_T"] = puid_eff_T
        jets["oneminuspuid_eff_T"] = 1.0-puid_eff_T
        jets["puid_sf_T"] = puid_sf_T
        jets["eff_sf_T"] = puid_sf_T * puid_eff_T
        jets["oneminus_eff_sf_T"] = 1.0-(puid_sf_T * puid_eff_T)
        pMC_passed_T = pd.Series(1, index=range(0, numevents))
        pMC_failed_T = pd.Series(1, index=range(0, numevents))
        pMC_passed_bare_T = jets[jets_passed_T==True].groupby('entry').puid_eff_T.prod()
        pMC_failed_bare_T =(jets.loc[jets_failed_T==True].groupby('entry').oneminuspuid_eff_T).prod()
        pMC_passed_T = pd.Series(1, index=range(0, numevents))
        pMC_failed_T = pd.Series(1, index=range(0, numevents))
        pMC_passed_T.update(pMC_passed_bare_T)
        pMC_failed_T.update(pMC_failed_bare_T)
        pSF_T = pd.Series(1, index=range(0, numevents))
        pfailSF_T = pd.Series(1, index=range(0, numevents))
        pSF_bare_T = jets[jets_passed_T==True].groupby('entry').puid_sf_T.prod()
        pfailSF_bare_T = (jets[jets_failed_T==True].groupby('entry').oneminus_eff_sf_T).prod()
        pSF_T.update(pSF_bare_T)
        pfailSF_T.update(pfailSF_bare_T)


        pMC_T = pMC_passed_T * pMC_failed_T
        pData_T = (
            pMC_passed_T
            * pSF_T
            * pfailSF_T
        )
        
        # merge jetpuid for loose and tight
        pMC = pMC_L * pMC_T
        pData = pData_L * pData_T 
        
        #print(pMC_passed)
        #print(pData)
        puid_weight = np.ones(numevents)
        puid_weight[pMC != 0] = np.divide(pData[pMC != 0], pMC[pMC != 0])
    else:
        wp_dict = {"loose": "L", "medium": "M", "tight": "T"}
        wp = wp_dict[jet_puid_opt]
        h_eff_name = f"h2_eff_mc{yearname}_L"
        h_sf_name = f"h2_eff_sf{yearname}_L"
        jetpt = jets[pt_name].values
        jeteta = jets.eta.values
        puid_eff = evaluator[h_eff_name](jetpt, jeteta)
        puid_sf = evaluator[h_sf_name](jetpt, jeteta)
        jets["puid_eff"] = puid_eff
        jets["oneminuspuid_eff"] = 1.0-puid_eff
        jets["puid_sf"] = puid_sf
        jets["eff_sf"] = puid_sf * puid_eff
        jets["oneminus_eff_sf"] = 1.0-(puid_sf * puid_eff)
        #with open(f'puideff.txt', 'w') as f:
            #print(jets["puid_eff"], file=f)
        jets_passed = (jets[pt_name] > 25) & (jets[pt_name] < 50) & jet_puid
        jets_failed = (jets[pt_name] > 25) & (jets[pt_name] < 50) & (~jet_puid)
        #with open(f'jetspass.txt', 'w') as f:
            #print(jets_passed, file=f) 
        #with open(f'jetsfail.txt', 'w') as f:
            #print(jets_failed, file=f) 
        pMC_failed = np.ones(numevents)
        pMC_passed_bare = jets[jets_passed==True].groupby('entry').puid_eff.prod()
        pMC_failed_bare =(jets.loc[jets_failed==True].groupby('entry').oneminuspuid_eff).prod()
        pMC_passed = pd.Series(1, index=range(0, numevents))
        pMC_failed = pd.Series(1, index=range(0, numevents))
        pMC_passed.update(pMC_passed_bare)
        pMC_failed.update(pMC_failed_bare)
        pSF = pd.Series(1, index=range(0, numevents))
        pfailSF = pd.Series(1, index=range(0, numevents))
        pSF_bare = jets[jets_passed==True].groupby('entry').puid_sf.prod()
        pfailSF_bare = (jets[jets_failed==True].groupby('entry').oneminus_eff_sf).prod()
        pSF.update(pSF_bare)
        pfailSF.update(pfailSF_bare)
        
        #print(pMC_failed)
        pMC = pMC_passed * pMC_failed

        pData = (
            pMC_passed
            * pSF
            * pfailSF
        )
        #print(pMC_passed)
        #print(pData)
        puid_weight = np.ones(numevents)
        puid_weight[pMC != 0] = np.divide(pData[pMC != 0], pMC[pMC != 0])
        #print(puid_weight)
    return puid_weight