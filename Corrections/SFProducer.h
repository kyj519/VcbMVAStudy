//
//
//
//
//
#include <string>
#include <iostream>
#include <algorithm>
#include "TFile.h"
#include "TH2F.h"

using namespace std;

class SFProducer
{

public:
  SFProducer(string, string, string);

  void LoadSF();

  float GetSF(int, float, int, int, int);
  float GetSF_TTLJ(int, float, int, int, int);
  float GetSF_none_TTLJ(int n_jets, float ht);

  TFile *m_SFFile = nullptr;
  TH2F *m_SFHist = nullptr;
  map<string, TH2F *> map_Hist_SF;

private:
  string m_eraName;
  string m_sfFileName;
  string m_sampleName;
};

SFProducer::SFProducer(string _eraName, string _sfFileName, string _sample)
{
  cout << "SFProducerPUJetId::SFProducerPUJetId()::In Constructor" << endl;
  m_eraName = _eraName;
  m_sfFileName = _sfFileName;
  m_sampleName = _sample;
}

void SFProducer::LoadSF()
{
  cout << "SFProducerPUJetId::LoadSF()::Load SF for era = " << m_eraName << ", sample = " << m_sampleName << endl;

  //
  // Load SF
  //
  m_SFFile = new TFile(m_sfFileName.c_str(), "READ");
  if (m_sampleName == "TTLJ")
  {
    string histName_2 = m_sampleName + "_2/Ratio_" + m_sampleName + "_2_B_Tag_Nominal";
    string histName_4 = m_sampleName + "_4/Ratio_" + m_sampleName + "_4_B_Tag_Nominal";
    string histName_45 = m_sampleName + "_45/Ratio_" + m_sampleName + "_45_B_Tag_Nominal";
    string histName_BB_2 = m_sampleName + "_BB_2/Ratio_" + m_sampleName + "_BB_2_B_Tag_Nominal";
    string histName_BB_4 = m_sampleName + "_BB_4/Ratio_" + m_sampleName + "_BB_4_B_Tag_Nominal";
    string histName_BB_45 = m_sampleName + "_BB_45/Ratio_" + m_sampleName + "_BB_45_B_Tag_Nominal";
    string histName_CC_2 = m_sampleName + "_CC_2/Ratio_" + m_sampleName + "_CC_2_B_Tag_Nominal";
    string histName_CC_4 = m_sampleName + "_CC_4/Ratio_" + m_sampleName + "_CC_4_B_Tag_Nominal";
    string histName_CC_45 = m_sampleName + "_CC_45/Ratio_" + m_sampleName + "_CC_45_B_Tag_Nominal";

    map_Hist_SF.insert({"002", (TH2F *)m_SFFile->Get(histName_2.c_str())});
    map_Hist_SF.insert({"004", (TH2F *)m_SFFile->Get(histName_4.c_str())});
    map_Hist_SF.insert({"0045", (TH2F *)m_SFFile->Get(histName_45.c_str())});
    map_Hist_SF.insert({"102", (TH2F *)m_SFFile->Get(histName_BB_2.c_str())});
    map_Hist_SF.insert({"104", (TH2F *)m_SFFile->Get(histName_BB_4.c_str())});
    map_Hist_SF.insert({"1045", (TH2F *)m_SFFile->Get(histName_BB_45.c_str())});
    map_Hist_SF.insert({"012", (TH2F *)m_SFFile->Get(histName_CC_2.c_str())});
    map_Hist_SF.insert({"014", (TH2F *)m_SFFile->Get(histName_CC_4.c_str())});
    map_Hist_SF.insert({"0145", (TH2F *)m_SFFile->Get(histName_CC_45.c_str())});
  }
  else
  {
    string histName = m_sampleName + "/Ratio_" + m_sampleName + "_B_Tag_Nominal";
    m_SFHist = (TH2F *)m_SFFile->Get(histName.c_str());
  }
}

//
//
//
float SFProducer::GetSF(int n_jets, float ht, int isBB = 0, int isCC = 0, int whatMode = 0)
{
  if (m_sampleName == "TTLJ")
  {
    return GetSF_TTLJ(n_jets, ht, isBB, isCC, whatMode);
  }
  else
    return GetSF_none_TTLJ(n_jets, ht);
}

float SFProducer::GetSF_TTLJ(int n_jets, float ht, int isBB, int isCC, int whatMode)
{
  float sf = 1.;

  if (n_jets < 4)
    n_jets = 4;
  else if (n_jets >= 30)
    n_jets = 30;

  if (ht < 80.)
    ht = 80.;
  else if (ht >= 1000)
    ht = 999.9;
  // cout << "isBB is " << isBB << " isCC is " << isCC << " whatMode" << whatMode << endl;
  string desiredKey = std::to_string(isBB) + std::to_string(isCC) + std::to_string(whatMode);
  m_SFHist = map_Hist_SF[desiredKey];
  // cout << m_SFHist->GetName() << endl;
  sf = m_SFHist->GetBinContent(m_SFHist->FindBin(n_jets, ht));
  return sf;
}

float SFProducer::GetSF_none_TTLJ(int n_jets, float ht)
{
  float sf = 1.;

  if (n_jets < 4)
    n_jets = 4;
  else if (n_jets >= 30)
    n_jets = 30;

  if (ht < 80.)
    ht = 80.;
  else if (ht >= 1000)
    ht = 999.9;

  sf = m_SFHist->GetBinContent(m_SFHist->FindBin(n_jets, ht));
  return sf;
}