#ifndef JET_SF_PRODUCER_H
#define JET_SF_PRODUCER_H

#include <string>
#include <memory>
#include <map>
#include <vector>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <ROOT/RVec.hxx>
using ROOT::VecOps::RVec;

#include "TFile.h"
#include "TH2D.h"

class JetSFProducer
{
public:
  JetSFProducer(std::string era,
                std::string file,
                std::string sample,
                std::string tagKind,
                int whatMode,
                bool useAbsEta = false)
      : m_era(std::move(era)),
        m_fileName(std::move(file)),
        m_sample(std::move(sample)),
        m_tagKind(std::move(tagKind)),
        m_whatMode(whatMode),
        m_useAbsEta(useAbsEta) {}

  JetSFProducer(std::string era,
                std::string file,
                std::string sample,
                std::string tagKind,
                bool useAbsEta = true)
      : JetSFProducer(std::move(era), std::move(file), std::move(sample),
                      std::move(tagKind), -1, useAbsEta) {}

  void Load()
  {
    std::cout << "[JetSFProducer] Load SFs from " << m_fileName
              << " (era=" << m_era << ", sample=" << m_sample
              << ", tag=" << m_tagKind
              << ", mode=" << m_whatMode << ")\n";

    std::unique_ptr<TFile> f(TFile::Open(m_fileName.c_str(), "READ"));
    if (!f || f->IsZombie())
    {
      throw std::runtime_error("Cannot open file: " + m_fileName);
    }

    // 경로/이름 만들기
    // 예) D/TTLJ_4/Ratio_D_TTLJ_4_B_Tag_Nominal_B
    const std::string modeSuffix = (m_whatMode >= 0) ? ("_" + std::to_string(m_whatMode)) : "";
    const std::string dir = "D/" + m_sample + modeSuffix;
    const std::string base = "Ratio_D_" + m_sample + modeSuffix + "_" + m_tagKind + "_Nominal_";

    // B, C, L 3개를 로드
    m_h["B"] = fetchAndDetachTH2D(f.get(), dir + "/" + base + "B");
    m_h["C"] = fetchAndDetachTH2D(f.get(), dir + "/" + base + "C");
    m_h["L"] = fetchAndDetachTH2D(f.get(), dir + "/" + base + "L");

    // 축 범위 저장 (clamp에 사용)
    // 세 히스토의 binning이 같다고 가정
    if (m_h["B"])
      cacheAxisRanges(m_h["B"]);
    else if (m_h["C"])
      cacheAxisRanges(m_h["C"]);
    else if (m_h["L"])
      cacheAxisRanges(m_h["L"]);
    else
    {
      throw std::runtime_error("No valid histograms loaded.");
    }
  }

  /// std::vector 버전
  float GetEventSF(const std::vector<float> &pts,
                   const std::vector<float> &etas,
                   const std::vector<int> &flavors) const
  {
    const size_t n = std::min({pts.size(), etas.size(), flavors.size()});
    float sf = 1.f;
    for (size_t i = 0; i < n; ++i)
    {
      sf *= GetJetSF(pts[i], etas[i], flavors[i]);
    }
    return sf;
  }

  float GetEventSF(const ROOT::RVec<float> &pts,
                   const ROOT::RVec<float> &etas,
                   const ROOT::RVec<int> &flavors) const
  {
    const size_t n = std::min({pts.size(), etas.size(), flavors.size()});
    float sf = 1.f;
    for (size_t i = 0; i < n; ++i)
    {
      sf *= GetJetSF(pts[i], etas[i], flavors[i]);
    }
    return sf;
  }

  /// 필요 시 개별 젯 SF만 계산
  float GetJetSF(float pt, float eta, int flavor) const
  {
    const char fl = flavorLetter(flavor);
    auto it = m_h.find(std::string(1, fl));
    if (it == m_h.end() || !it->second)
    {
      // fallback: 없으면 runtime 에러
      throw std::runtime_error("Jet flavor " + std::string(1, fl) + " not found in histograms.");
    }

    const float x = clamp(pt, m_xmin, m_xmax * 0.9999f);
    const float yraw = m_useAbsEta ? std::fabs(eta) : eta;
    const float y = clamp(yraw, m_ymin, m_ymax * 0.9999f);

    const int bin = it->second->FindBin(x, y);
    return it->second->GetBinContent(bin);
  }

private:
  static TH2D *fetchAndDetachTH2D(TFile *f, const std::string &name)
  {
    TH2D *h = dynamic_cast<TH2D *>(f->Get(name.c_str()));
    if (!h)
    {
      std::cerr << "[JetSFProducer] WARNING: hist not found: " << name << "\n";
      return nullptr;
    }
    TH2D *hc = (TH2D *)h->Clone();
    hc->SetDirectory(nullptr);
    return hc;
  }

  static inline char flavorLetter(int hadFlavor)
  {
    const int af = std::abs(hadFlavor);
    if (af == 5)
      return 'B';
    if (af == 4)
      return 'C';
    return 'L';
  }

  static inline float clamp(float v, float lo, float hi)
  {
    return std::max(lo, std::min(v, hi));
  }

  void cacheAxisRanges(TH2D *h)
  {
    if (!h)
      return;
    m_xmin = h->GetXaxis()->GetXmin();
    m_xmax = h->GetXaxis()->GetXmax();
    m_ymin = h->GetYaxis()->GetXmin();
    m_ymax = h->GetYaxis()->GetXmax();
  }

private:
  std::string m_era;
  std::string m_fileName;
  std::string m_sample;
  std::string m_tagKind; // "B_Tag" or "C_Tag"
  int m_whatMode = -1;
  bool m_useAbsEta = true;

  std::map<std::string, TH2D *> m_h; // "B","C","L"

  // axis cache
  float m_xmin = 0, m_xmax = 0;
  float m_ymin = 0, m_ymax = 0;
};

#endif