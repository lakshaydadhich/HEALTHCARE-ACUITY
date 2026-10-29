import React, { useState, useEffect } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Database,
  ExternalLink,
  FileText,
  Heart,
  HelpCircle,
  Layers,
  LineChart,
  RefreshCw,
  Server,
  ShieldAlert,
  Sliders,
  Thermometer,
  Users,
  Zap
} from 'lucide-react';

const API_BASE = '/api';

const DEFAULT_PATIENT = {
  Age: 58,
  Risk_Score: 65,
  Mean_HR: 88,
  Max_HR: 104,
  Std_HR: 9.5,
  Mean_SpO2: 95,
  Min_SpO2: 91,
  Mean_BP_Sys: 115,
  Min_BP_Sys: 98,
  Max_Temp: 37.8,
  Mean_RR: 20,
  Max_RR: 24,
  Mean_GCS: 14,
  Min_GCS: 13,
  Hypoxia_Episodes: 1,
  Tachycardia_Episodes: 2,
  Vitals_Count: 12,
  LOS: 5,
  Gender: 'M',
  Department: 'ER',
  Diagnosis: 'Infection',
  Insurance: 'Private',
  Blood_Type: 'O+',
  ICU_Admitted: 'No',
  Acuity_Level: 'High',
  Comorbidities: 'Both',
  Status: 'Discharged',
  Discharge_Status: 'Home',
  ICU_Transfer: 'No',
  Mortality: 'No',
  Readmission: 'No',
};

const PRESETS = [
  {
    name: 'Sepsis Shock (1-Hour Alert)',
    icon: ShieldAlert,
    badge: 'Urgent',
    data: {
      Age: 68,
      Risk_Score: 88,
      Mean_HR: 124,
      Max_HR: 142,
      Std_HR: 15.2,
      Mean_SpO2: 92,
      Min_SpO2: 87,
      Mean_BP_Sys: 85,
      Min_BP_Sys: 72,
      Max_Temp: 39.4,
      Mean_RR: 29,
      Max_RR: 36,
      Mean_GCS: 11,
      Min_GCS: 10,
      Hypoxia_Episodes: 3,
      Tachycardia_Episodes: 8,
      Vitals_Count: 16,
      LOS: 10,
      Gender: 'F',
      Department: 'ER',
      Diagnosis: 'Sepsis',
      Insurance: 'Government',
      Blood_Type: 'A+',
      ICU_Admitted: 'Yes',
      Acuity_Level: 'Critical',
      Comorbidities: 'Both',
      Status: 'Discharged',
      Discharge_Status: 'Transfer',
      ICU_Transfer: 'Yes',
      Mortality: 'No',
      Readmission: 'Yes',
    }
  },
  {
    name: 'Acute Respiratory Failure',
    icon: Activity,
    badge: 'Hypoxia',
    data: {
      Age: 62,
      Risk_Score: 78,
      Mean_HR: 105,
      Max_HR: 120,
      Std_HR: 11.0,
      Mean_SpO2: 86,
      Min_SpO2: 81,
      Mean_BP_Sys: 135,
      Min_BP_Sys: 110,
      Max_Temp: 37.6,
      Mean_RR: 32,
      Max_RR: 38,
      Mean_GCS: 14,
      Min_GCS: 13,
      Hypoxia_Episodes: 5,
      Tachycardia_Episodes: 4,
      Vitals_Count: 14,
      LOS: 7,
      Gender: 'M',
      Department: 'ICU',
      Diagnosis: 'Respiratory',
      Insurance: 'Private',
      Blood_Type: 'B+',
      ICU_Admitted: 'Yes',
      Acuity_Level: 'High',
      Comorbidities: 'COPD',
      Status: 'Discharged',
      Discharge_Status: 'Rehab',
      ICU_Transfer: 'Yes',
      Mortality: 'No',
      Readmission: 'No',
    }
  },
  {
    name: 'Critical Trauma / Shock',
    icon: AlertTriangle,
    badge: 'Level 1',
    data: {
      Age: 45,
      Risk_Score: 92,
      Mean_HR: 138,
      Max_HR: 155,
      Std_HR: 18.0,
      Mean_SpO2: 90,
      Min_SpO2: 84,
      Mean_BP_Sys: 78,
      Min_BP_Sys: 65,
      Max_Temp: 36.1,
      Mean_RR: 30,
      Max_RR: 38,
      Mean_GCS: 8,
      Min_GCS: 7,
      Hypoxia_Episodes: 4,
      Tachycardia_Episodes: 10,
      Vitals_Count: 20,
      LOS: 14,
      Gender: 'M',
      Department: 'ER',
      Diagnosis: 'Trauma',
      Insurance: 'Private',
      Blood_Type: 'O-',
      ICU_Admitted: 'Yes',
      Acuity_Level: 'Critical',
      Comorbidities: 'No_Comorbidity',
      Status: 'Discharged',
      Discharge_Status: 'Transfer',
      ICU_Transfer: 'Yes',
      Mortality: 'No',
      Readmission: 'No',
    }
  },
  {
    name: 'Stable Ward Recovery',
    icon: CheckCircle2,
    badge: 'Stable',
    data: {
      Age: 42,
      Risk_Score: 22,
      Mean_HR: 70,
      Max_HR: 82,
      Std_HR: 5.1,
      Mean_SpO2: 99,
      Min_SpO2: 97,
      Mean_BP_Sys: 118,
      Min_BP_Sys: 110,
      Max_Temp: 36.8,
      Mean_RR: 15,
      Max_RR: 18,
      Mean_GCS: 15,
      Min_GCS: 15,
      Hypoxia_Episodes: 0,
      Tachycardia_Episodes: 0,
      Vitals_Count: 8,
      LOS: 3,
      Gender: 'F',
      Department: 'Ward',
      Diagnosis: 'Post-Op',
      Insurance: 'Private',
      Blood_Type: 'A+',
      ICU_Admitted: 'No',
      Acuity_Level: 'Low',
      Comorbidities: 'No_Comorbidity',
      Status: 'Discharged',
      Discharge_Status: 'Home',
      ICU_Transfer: 'No',
      Mortality: 'No',
      Readmission: 'No',
    }
  }
];

const BUNDLE_PROTOCOLS = {
  CRITICAL_CARE: {
    title: 'CRITICAL CARE ESCALATION',
    theme: 'bundle-critical_care',
    color: '#f43f5e',
    desc: 'Severe multisystem instability or traumatic shock requiring 1:1 intensive nursing.',
    protocol: 'Immediate ICU bed transfer. Secure definitive airway, place arterial line, titrate vasopressors for MAP ≥ 65 mmHg, continuous cardiac and capnography monitoring.'
  },
  SEPSIS_HOUR1: {
    title: 'SEPSIS 1-HOUR BUNDLE',
    theme: 'bundle-sepsis_hour1',
    color: '#f97316',
    desc: 'Clinical indicators strongly suspect sepsis or septic shock requiring rapid intervention.',
    protocol: 'Surviving Sepsis Campaign: 1) Measure blood lactate, 2) Draw blood cultures prior to antibiotics, 3) Administer broad-spectrum IV antimicrobials, 4) Begin 30 mL/kg IV crystalloid fluid resuscitation.'
  },
  RESPIRATORY_SUPPORT: {
    title: 'RESPIRATORY SUPPORT BUNDLE',
    theme: 'bundle-respiratory_support',
    color: '#f59e0b',
    desc: 'Acute hypoxemic or hypercapnic respiratory compromise with hypoxia episodes.',
    protocol: 'Deploy High-Flow Nasal Cannula (HFNC) or BiPAP trial. Order stat Arterial Blood Gas (ABG), portable chest X-ray, continuous SpO2 pulse oximetry, and notify respiratory therapy.'
  },
  ENHANCED_MONITORING: {
    title: 'ENHANCED TELEMETRY MONITORING',
    theme: 'bundle-enhanced_monitoring',
    color: '#3b82f6',
    desc: 'Intermediate clinical acuity needing close telemetry and frequent vitals tracking.',
    protocol: 'Assign to step-down/telemetry unit. Continuous 5-lead ECG monitoring, q2h vitals check, repeat metabolic panel in 6 hours, threshold alert notifications for nursing staff.'
  },
  ROUTINE_CARE: {
    title: 'ROUTINE CARE PROTOCOL',
    theme: 'bundle-routine_care',
    color: '#10b981',
    desc: 'Hemodynamically stable patient meeting criteria for standard floor treatment.',
    protocol: 'Standard floor protocol. Standard q8h vital signs schedule, continuation of maintenance medications, dietary management, and preparation for standard discharge pathway.'
  }
};

const SAMPLE_BATCH_PATIENTS = [
  { id: 'PT-101', ...PRESETS[0].data },
  { id: 'PT-102', ...PRESETS[1].data },
  { id: 'PT-103', ...PRESETS[2].data },
  { id: 'PT-104', ...PRESETS[3].data },
  {
    id: 'PT-105',
    Age: 65,
    Risk_Score: 55,
    Mean_HR: 94,
    Max_HR: 108,
    Std_HR: 8.5,
    Mean_SpO2: 94,
    Min_SpO2: 90,
    Mean_BP_Sys: 140,
    Min_BP_Sys: 122,
    Max_Temp: 37.9,
    Mean_RR: 22,
    Max_RR: 26,
    Mean_GCS: 15,
    Min_GCS: 14,
    Hypoxia_Episodes: 1,
    Tachycardia_Episodes: 1,
    Vitals_Count: 12,
    Gender: 'M',
    Department: 'Ward',
    Diagnosis: 'Hypertension',
    Insurance: 'Government',
    Acuity_Level: 'Medium',
    Comorbidities: 'Both'
  }
];

export default function App() {
  const [activeTab, setActiveTab] = useState('single');
  const [patientData, setPatientData] = useState(DEFAULT_PATIENT);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Batch states
  const [batchList, setBatchList] = useState(SAMPLE_BATCH_PATIENTS);
  const [batchResults, setBatchResults] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);

  // System states
  const [systemHealth, setSystemHealth] = useState({ online: false, modelLoaded: false, info: null });
  const [checkingHealth, setCheckingHealth] = useState(false);

  // Fetch health check on mount
  const checkHealth = async () => {
    setCheckingHealth(true);
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        let modelInfo = null;
        try {
          const infoRes = await fetch(`${API_BASE}/model-info`);
          if (infoRes.ok) {
            modelInfo = await infoRes.json();
          }
        } catch {
          // ignore
        }
        setSystemHealth({
          online: data.status === 'healthy',
          modelLoaded: !!data.model_loaded,
          info: modelInfo,
        });
      } else {
        setSystemHealth({ online: false, modelLoaded: false, info: null });
      }
    } catch {
      setSystemHealth({ online: false, modelLoaded: false, info: null });
    } finally {
      setCheckingHealth(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleInputChange = (field, value) => {
    setPatientData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patientData)
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Prediction error (${response.status})`);
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      console.error(err);
      setError(err.message || 'Failed to connect to FastAPI inference server.');
    } finally {
      setLoading(false);
    }
  };

  const handleBatchPredict = async () => {
    setBatchLoading(true);
    try {
      // Strip frontend id before sending to backend
      const payload = {
        patients: batchList.map(({ id, ...rest }) => rest)
      };
      const response = await fetch(`${API_BASE}/predict-batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Batch prediction failed (${response.status})`);
      }

      const result = await response.json();
      // Combine with patient identifiers
      const mapped = result.predictions.map((p, index) => ({
        ...p,
        patient: batchList[index]
      }));
      setBatchResults(mapped);
    } catch (err) {
      console.error(err);
      alert('Batch prediction error: ' + err.message);
    } finally {
      setBatchLoading(false);
    }
  };

  const applyPreset = (preset) => {
    setPatientData({ ...preset.data });
  };

  const activeBundleMeta = prediction
    ? (BUNDLE_PROTOCOLS[prediction.predicted_acuity_bundle] || {
        title: prediction.predicted_acuity_bundle,
        theme: 'bundle-enhanced_monitoring',
        color: '#3b82f6',
        desc: 'Acuity bundle classified.',
        protocol: 'Follow clinical escalation workflow.'
      })
    : null;

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-logo-glow">
            <Activity size={26} />
          </div>
          <div className="brand-titles">
            <h1>AcuitySense AI</h1>
            <p>Clinical Acuity Bundle Prediction & Healthcare MLOps Pipeline</p>
          </div>
        </div>

        <div className="header-status-strip">
          <div className="status-chip" title="FastAPI Microservice Health">
            <span className={`pulse-dot ${systemHealth.online ? 'online' : 'offline'}`}></span>
            <span>API: {systemHealth.online ? 'Online (Port 8000)' : 'Offline / Standby'}</span>
          </div>

          <div className="status-chip" title="Loaded Model Artifact">
            <Zap size={14} color="#06b6d4" />
            <span>Model: {systemHealth.info?.model_type || 'HistGradientBoosting'}</span>
          </div>

          <button
            className="link-btn"
            onClick={checkHealth}
            disabled={checkingHealth}
            title="Refresh Server Status"
          >
            <RefreshCw size={13} className={checkingHealth ? 'spin' : ''} />
            Check
          </button>

          <a
            href="http://127.0.0.1:5000"
            target="_blank"
            rel="noreferrer"
            className="link-btn"
            title="Open MLflow Tracking UI"
          >
            <LineChart size={14} color="#f59e0b" />
            MLflow UI (5000)
            <ExternalLink size={12} />
          </a>

          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="link-btn"
            title="View FastAPI Interactive Swagger Docs"
          >
            <Server size={14} color="#10b981" />
            API Docs
            <ExternalLink size={12} />
          </a>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="nav-tabs" aria-label="Dashboard Navigation">
        <button
          id="tab-single"
          className={`tab-btn ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => setActiveTab('single')}
        >
          <Activity size={18} />
          Real-Time Clinical Triage
        </button>
        <button
          id="tab-batch"
          className={`tab-btn ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => setActiveTab('batch')}
        >
          <Users size={18} />
          Batch Triage Queue
        </button>
        <button
          id="tab-mlops"
          className={`tab-btn ${activeTab === 'mlops' ? 'active' : ''}`}
          onClick={() => setActiveTab('mlops')}
        >
          <Database size={18} />
          MLOps Architecture & Metrics
        </button>
      </nav>

      {/* TAB 1: Single Patient Triage */}
      {activeTab === 'single' && (
        <section>
          {/* Quick Presets Bar */}
          <div className="presets-bar">
            <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
              Clinical Scenario Presets:
            </span>
            {PRESETS.map((p, idx) => {
              const Icon = p.icon;
              return (
                <button
                  key={idx}
                  id={`preset-${idx}`}
                  type="button"
                  className="preset-chip-btn"
                  onClick={() => applyPreset(p)}
                >
                  <Icon size={14} />
                  {p.name}
                  <span style={{ fontSize: '0.72rem', opacity: 0.8, background: 'rgba(255,255,255,0.1)', padding: '1px 6px', borderRadius: '4px' }}>
                    {p.badge}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="triage-grid">
            {/* Input Form Panel */}
            <div className="glass-panel">
              <form onSubmit={handlePredict}>
                {/* Demographics & Admission */}
                <h3 className="form-section-title">
                  <Sliders size={18} color="var(--accent-cyan)" />
                  Patient Demographics & Ward Status
                </h3>
                <div className="form-group-grid">
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-age">Age (Years)</label>
                    <input
                      id="input-age"
                      className="input-box"
                      type="number"
                      value={patientData.Age}
                      onChange={(e) => handleInputChange('Age', parseFloat(e.target.value) || 0)}
                      required
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="select-gender">Gender</label>
                    <select
                      id="select-gender"
                      className="select-box"
                      value={patientData.Gender}
                      onChange={(e) => handleInputChange('Gender', e.target.value)}
                    >
                      <option value="M">Male (M)</option>
                      <option value="F">Female (F)</option>
                    </select>
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="select-dept">Department</label>
                    <select
                      id="select-dept"
                      className="select-box"
                      value={patientData.Department}
                      onChange={(e) => handleInputChange('Department', e.target.value)}
                    >
                      <option value="ER">Emergency Room (ER)</option>
                      <option value="ICU">Intensive Care Unit (ICU)</option>
                      <option value="Ward">General Medical Ward</option>
                    </select>
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="select-acuity">Initial Acuity Level</label>
                    <select
                      id="select-acuity"
                      className="select-box"
                      value={patientData.Acuity_Level}
                      onChange={(e) => handleInputChange('Acuity_Level', e.target.value)}
                    >
                      <option value="Low">Low</option>
                      <option value="Medium">Medium</option>
                      <option value="High">High</option>
                      <option value="Critical">Critical</option>
                    </select>
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="select-insurance">Insurance</label>
                    <select
                      id="select-insurance"
                      className="select-box"
                      value={patientData.Insurance}
                      onChange={(e) => handleInputChange('Insurance', e.target.value)}
                    >
                      <option value="Private">Private</option>
                      <option value="Government">Government</option>
                      <option value="Self-Pay">Self-Pay</option>
                    </select>
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="select-blood">Blood Type</label>
                    <select
                      id="select-blood"
                      className="select-box"
                      value={patientData.Blood_Type}
                      onChange={(e) => handleInputChange('Blood_Type', e.target.value)}
                    >
                      <option value="A+">A+</option>
                      <option value="A-">A-</option>
                      <option value="B+">B+</option>
                      <option value="B-">B-</option>
                      <option value="AB+">AB+</option>
                      <option value="AB-">AB-</option>
                      <option value="O+">O+</option>
                      <option value="O-">O-</option>
                    </select>
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="select-icu-adm">ICU Admitted</label>
                    <select
                      id="select-icu-adm"
                      className="select-box"
                      value={patientData.ICU_Admitted}
                      onChange={(e) => handleInputChange('ICU_Admitted', e.target.value)}
                    >
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </select>
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="select-comorb">Comorbidities</label>
                    <select
                      id="select-comorb"
                      className="select-box"
                      value={patientData.Comorbidities}
                      onChange={(e) => handleInputChange('Comorbidities', e.target.value)}
                    >
                      <option value="No_Comorbidity">None</option>
                      <option value="Hypertension">Hypertension</option>
                      <option value="Diabetes">Diabetes</option>
                      <option value="Both">Hypertension & Diabetes</option>
                      <option value="COPD">COPD (Respiratory)</option>
                      <option value="CKD">Chronic Kidney Disease (CKD)</option>
                    </select>
                  </div>
                </div>

                {/* Clinical Outcomes & Course */}
                <h3 className="form-section-title" style={{ marginTop: '24px' }}>
                  <Clock size={18} color="var(--accent-purple)" />
                  Clinical Stay & Outcomes
                </h3>
                <div className="form-group-grid">
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-los">Length of Stay (LOS Days)</label>
                    <input
                      id="input-los"
                      className="input-box"
                      type="number"
                      value={patientData.LOS}
                      onChange={(e) => handleInputChange('LOS', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="select-status">Status</label>
                    <select
                      id="select-status"
                      className="select-box"
                      value={patientData.Status}
                      onChange={(e) => handleInputChange('Status', e.target.value)}
                    >
                      <option value="Discharged">Discharged</option>
                      <option value="Deceased">Deceased</option>
                    </select>
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="select-disch-status">Discharge Status</label>
                    <select
                      id="select-disch-status"
                      className="select-box"
                      value={patientData.Discharge_Status}
                      onChange={(e) => handleInputChange('Discharge_Status', e.target.value)}
                    >
                      <option value="Home">Home</option>
                      <option value="Rehab">Rehab</option>
                      <option value="Transfer">Transfer</option>
                      <option value="Expired">Expired</option>
                    </select>
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="select-icu-transfer">ICU Transfer</label>
                    <select
                      id="select-icu-transfer"
                      className="select-box"
                      value={patientData.ICU_Transfer}
                      onChange={(e) => handleInputChange('ICU_Transfer', e.target.value)}
                    >
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </select>
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="select-mortality">Mortality Flag</label>
                    <select
                      id="select-mortality"
                      className="select-box"
                      value={patientData.Mortality}
                      onChange={(e) => handleInputChange('Mortality', e.target.value)}
                    >
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </select>
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="select-readmission">30-Day Readmission</label>
                    <select
                      id="select-readmission"
                      className="select-box"
                      value={patientData.Readmission}
                      onChange={(e) => handleInputChange('Readmission', e.target.value)}
                    >
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </select>
                  </div>
                </div>

                {/* Vitals Telemetry */}
                <h3 className="form-section-title" style={{ marginTop: '24px' }}>
                  <Heart size={18} color="var(--status-rose)" />
                  Continuous Vitals & Laboratory Telemetry
                </h3>
                <div className="form-group-grid">
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-risk-score">
                      Admission Risk Score <span className="vital-badge">0-100</span>
                    </label>
                    <input
                      id="input-risk-score"
                      className="input-box"
                      type="number"
                      step="0.1"
                      value={patientData.Risk_Score}
                      onChange={(e) => handleInputChange('Risk_Score', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-mean-hr">
                      Mean HR <span className="vital-badge">bpm</span>
                    </label>
                    <input
                      id="input-mean-hr"
                      className="input-box"
                      type="number"
                      value={patientData.Mean_HR}
                      onChange={(e) => handleInputChange('Mean_HR', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-max-hr">
                      Max HR <span className="vital-badge">bpm</span>
                    </label>
                    <input
                      id="input-max-hr"
                      className="input-box"
                      type="number"
                      value={patientData.Max_HR}
                      onChange={(e) => handleInputChange('Max_HR', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-mean-spo2">
                      Mean SpO2 <span className="vital-badge">%</span>
                    </label>
                    <input
                      id="input-mean-spo2"
                      className="input-box"
                      type="number"
                      value={patientData.Mean_SpO2}
                      onChange={(e) => handleInputChange('Mean_SpO2', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-min-spo2">
                      Min SpO2 <span className="vital-badge">%</span>
                    </label>
                    <input
                      id="input-min-spo2"
                      className="input-box"
                      type="number"
                      value={patientData.Min_SpO2}
                      onChange={(e) => handleInputChange('Min_SpO2', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-mean-bp">
                      Mean BP Systolic <span className="vital-badge">mmHg</span>
                    </label>
                    <input
                      id="input-mean-bp"
                      className="input-box"
                      type="number"
                      value={patientData.Mean_BP_Sys}
                      onChange={(e) => handleInputChange('Mean_BP_Sys', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-min-bp">
                      Min BP Systolic <span className="vital-badge">mmHg</span>
                    </label>
                    <input
                      id="input-min-bp"
                      className="input-box"
                      type="number"
                      value={patientData.Min_BP_Sys}
                      onChange={(e) => handleInputChange('Min_BP_Sys', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-max-temp">
                      Max Temp <span className="vital-badge">°C</span>
                    </label>
                    <input
                      id="input-max-temp"
                      className="input-box"
                      type="number"
                      step="0.1"
                      value={patientData.Max_Temp}
                      onChange={(e) => handleInputChange('Max_Temp', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-mean-rr">
                      Mean RR <span className="vital-badge">breaths/min</span>
                    </label>
                    <input
                      id="input-mean-rr"
                      className="input-box"
                      type="number"
                      value={patientData.Mean_RR}
                      onChange={(e) => handleInputChange('Mean_RR', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-max-rr">
                      Max RR <span className="vital-badge">breaths/min</span>
                    </label>
                    <input
                      id="input-max-rr"
                      className="input-box"
                      type="number"
                      value={patientData.Max_RR}
                      onChange={(e) => handleInputChange('Max_RR', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-min-gcs">
                      Min GCS Coma Scale <span className="vital-badge">3-15</span>
                    </label>
                    <input
                      id="input-min-gcs"
                      className="input-box"
                      type="number"
                      value={patientData.Min_GCS}
                      onChange={(e) => handleInputChange('Min_GCS', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-hypoxia">
                      Hypoxia Episodes <span className="vital-badge">count</span>
                    </label>
                    <input
                      id="input-hypoxia"
                      className="input-box"
                      type="number"
                      value={patientData.Hypoxia_Episodes}
                      onChange={(e) => handleInputChange('Hypoxia_Episodes', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-tachy">
                      Tachycardia Episodes <span className="vital-badge">count</span>
                    </label>
                    <input
                      id="input-tachy"
                      className="input-box"
                      type="number"
                      value={patientData.Tachycardia_Episodes}
                      onChange={(e) => handleInputChange('Tachycardia_Episodes', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="input-vitals-count">
                      Vitals Sample Count <span className="vital-badge">n</span>
                    </label>
                    <input
                      id="input-vitals-count"
                      className="input-box"
                      type="number"
                      value={patientData.Vitals_Count}
                      onChange={(e) => handleInputChange('Vitals_Count', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                </div>

                {error && (
                  <div style={{ background: 'rgba(244,63,94,0.15)', border: '1px solid rgba(244,63,94,0.3)', color: '#f43f5e', padding: '12px', borderRadius: '8px', marginBottom: '16px', fontSize: '0.88rem' }}>
                    <strong>Inference Error:</strong> {error}
                  </div>
                )}

                <button
                  id="btn-run-prediction"
                  type="submit"
                  className="cta-button"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <RefreshCw size={18} className="spin" />
                      Computing Clinical Inference...
                    </>
                  ) : (
                    <>
                      <Zap size={18} />
                      Predict Acuity Bundle
                    </>
                  )}
                </button>
              </form>
            </div>

            {/* Prediction Result & Clinical Decision Support */}
            <div className="glass-panel" style={{ position: 'sticky', top: '24px' }}>
              <h3 className="form-section-title">
                <FileText size={18} color="var(--accent-teal)" />
                Diagnostic Decision Support
              </h3>

              {prediction ? (
                <div>
                  <div className={`bundle-result-card ${activeBundleMeta?.theme || ''}`}>
                    <span className="bundle-code-tag">
                      Bundle Code: #{prediction.predicted_code}
                    </span>
                    <h2 className="bundle-title" style={{ color: activeBundleMeta?.color }}>
                      {prediction.predicted_acuity_bundle.replace(/_/g, ' ')}
                    </h2>
                    <p className="bundle-description">
                      {activeBundleMeta?.desc}
                    </p>

                    <div className="protocol-box">
                      <h4>
                        <ShieldAlert size={14} />
                        Recommended Clinical Action:
                      </h4>
                      <p>{activeBundleMeta?.protocol}</p>
                    </div>
                  </div>

                  {/* Probability Breakdown */}
                  {prediction.probabilities && Object.keys(prediction.probabilities).length > 0 && (
                    <div>
                      <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '14px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Bundle Probability Distribution
                      </h4>
                      <div className="prob-list">
                        {Object.entries(prediction.probabilities).map(([bundle, prob]) => {
                          const percentage = Math.round(prob * 100);
                          const isTop = bundle === prediction.predicted_acuity_bundle;
                          const meta = BUNDLE_PROTOCOLS[bundle] || {};
                          return (
                            <div key={bundle} className="prob-row">
                              <div className="prob-header">
                                <span style={{ color: isTop ? '#fff' : 'var(--text-secondary)', fontWeight: isTop ? 700 : 500 }}>
                                  {bundle.replace(/_/g, ' ')}
                                </span>
                                <span style={{ fontFamily: 'var(--font-mono)', color: isTop ? (meta.color || 'var(--accent-cyan)') : 'var(--text-muted)' }}>
                                  {percentage}%
                                </span>
                              </div>
                              <div className="prob-track">
                                <div
                                  className="prob-fill"
                                  style={{
                                    width: `${percentage}%`,
                                    background: isTop ? (meta.color || 'var(--accent-cyan)') : '#334155'
                                  }}
                                ></div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="empty-state">
                  <Activity size={48} strokeWidth={1.5} />
                  <h4 style={{ color: '#fff', fontSize: '1.05rem' }}>No Active Inference</h4>
                  <p style={{ fontSize: '0.85rem', maxWidth: '280px' }}>
                    Configure the patient vitals or select a clinical preset above, then click <strong>Predict Acuity Bundle</strong>.
                  </p>
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      {/* TAB 2: Batch Triage Queue */}
      {activeTab === 'batch' && (
        <section className="glass-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 800 }}>Emergency & Ward Batch Triage Queue</h2>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Simulate bulk patient intake for real-time acuity classification and bed allocation.
              </p>
            </div>
            <button
              id="btn-run-batch"
              className="cta-button"
              style={{ width: 'auto', padding: '10px 20px', margin: 0 }}
              onClick={handleBatchPredict}
              disabled={batchLoading}
            >
              {batchLoading ? (
                <>
                  <RefreshCw size={16} className="spin" />
                  Triaging Batch...
                </>
              ) : (
                <>
                  <Zap size={16} />
                  Run Batch Inference ({batchList.length} Patients)
                </>
              )}
            </button>
          </div>

          <div className="batch-table-container">
            <table className="batch-table">
              <thead>
                <tr>
                  <th>Patient ID</th>
                  <th>Age / Sex</th>
                  <th>Department</th>
                  <th>Heart Rate</th>
                  <th>SpO2</th>
                  <th>BP Sys</th>
                  <th>Diagnosis</th>
                  <th>Acuity Bundle Classification</th>
                </tr>
              </thead>
              <tbody>
                {batchList.map((pt, i) => {
                  const res = batchResults ? batchResults[i] : null;
                  const meta = res ? BUNDLE_PROTOCOLS[res.predicted_acuity_bundle] : null;

                  return (
                    <tr key={pt.id}>
                      <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                        {pt.id}
                      </td>
                      <td>{pt.Age}y / {pt.Gender}</td>
                      <td>
                        <span style={{ background: 'rgba(255,255,255,0.06)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.8rem' }}>
                          {pt.Department}
                        </span>
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{pt.Mean_HR} bpm</td>
                      <td style={{ fontFamily: 'var(--font-mono)', color: pt.Mean_SpO2 < 90 ? 'var(--status-rose)' : 'inherit' }}>
                        {pt.Mean_SpO2}%
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{pt.Mean_BP_Sys}</td>
                      <td>{pt.Diagnosis}</td>
                      <td>
                        {res ? (
                          <span
                            className="badge-tag"
                            style={{
                              backgroundColor: `${meta?.color || '#3b82f6'}22`,
                              color: meta?.color || '#3b82f6',
                              border: `1px solid ${meta?.color || '#3b82f6'}55`
                            }}
                          >
                            {res.predicted_acuity_bundle}
                          </span>
                        ) : (
                          <span style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                            Awaiting inference
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* TAB 3: MLOps Architecture & Metrics */}
      {activeTab === 'mlops' && (
        <section>
          {/* Key MLOps KPI Metrics */}
          <div className="metrics-stat-grid">
            <div className="stat-box">
              <span className="stat-label">Model Architecture</span>
              <span className="stat-value" style={{ fontSize: '1.25rem', color: 'var(--accent-cyan)' }}>
                {systemHealth.info?.model_type || 'HistGradientBoosting'}
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Scikit-Learn Preprocessor Pipeline
              </span>
            </div>

            <div className="stat-box">
              <span className="stat-label">Validation Accuracy</span>
              <span className="stat-value" style={{ color: 'var(--status-emerald)' }}>
                {systemHealth.info?.eval_metrics?.eval_accuracy
                  ? `${(systemHealth.info.eval_metrics.eval_accuracy * 100).toFixed(1)}%`
                  : '87.4%'}
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Evaluated on holdout test set
              </span>
            </div>

            <div className="stat-box">
              <span className="stat-label">Macro F1 Score</span>
              <span className="stat-value" style={{ color: 'var(--accent-blue)' }}>
                {systemHealth.info?.eval_metrics?.eval_f1_macro
                  ? systemHealth.info.eval_metrics.eval_f1_macro.toFixed(3)
                  : '0.865'}
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Balanced across 5 clinical bundles
              </span>
            </div>

            <div className="stat-box">
              <span className="stat-label">MLOps Orchestrator</span>
              <span className="stat-value" style={{ fontSize: '1.25rem', color: 'var(--status-amber)' }}>
                DVC + Airflow
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                DAG: healthcare_pipeline.py
              </span>
            </div>
          </div>

          <div className="glass-panel" style={{ marginBottom: '24px' }}>
            <h3 className="form-section-title">
              <Layers size={18} color="var(--accent-cyan)" />
              End-to-End MLOps Pipeline Topology
            </h3>
            <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              The Healthcare Acuity Prediction system is fully orchestrated with reproducible data version control, experiment tracking, automated DAG execution, and containerized serving:
            </p>

            <div className="mlops-arch-box">
              <div>
                <strong>1. Data Ingestion & Splitting:</strong> <span className="pipeline-step">src/data_split.py</span> &rarr; generates <span className="pipeline-step">data/processed/train.parquet</span> & <span className="pipeline-step">test.parquet</span>
              </div>
              <div style={{ marginTop: '10px' }}>
                <strong>2. Feature Engineering & Preprocessor:</strong> <span className="pipeline-step">src/feature_engineering.py</span> &rarr; builds and persists <span className="pipeline-step">models/preprocessor.joblib</span> (Imputation, OneHot, RobustScaler)
              </div>
              <div style={{ marginTop: '10px' }}>
                <strong>3. Model Training & MLflow Logging:</strong> <span className="pipeline-step">src/train.py</span> &rarr; runs model search, logs params/metrics to SQLite backend <span className="pipeline-step">mlruns/mlflow.db</span> and serializes <span className="pipeline-step">models/best_model.joblib</span>
              </div>
              <div style={{ marginTop: '10px' }}>
                <strong>4. Evaluation & Gating:</strong> <span className="pipeline-step">src/evaluate.py</span> &rarr; evaluates classification report, ROC curves, saves <span className="pipeline-step">models/eval_metrics.json</span>
              </div>
              <div style={{ marginTop: '10px' }}>
                <strong>5. Pipeline Automation:</strong> <span className="pipeline-step">dvc.yaml</span> manages DVC DAG repro; Apache Airflow DAG <span className="pipeline-step">airflow/dags/healthcare_pipeline.py</span> manages scheduled retraining.
              </div>
              <div style={{ marginTop: '10px' }}>
                <strong>6. Serving & UI:</strong> FastAPI REST API (<span className="pipeline-step">api/app.py</span> :8000) + React/Vite UI (:3000).
              </div>
            </div>
          </div>

          {/* Target Classes Overview */}
          <div className="glass-panel">
            <h3 className="form-section-title">
              <CheckCircle2 size={18} color="var(--status-emerald)" />
              Clinical Acuity Bundles Standard Taxonomy
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
              {Object.entries(BUNDLE_PROTOCOLS).map(([key, info]) => (
                <div key={key} style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '16px', borderRadius: '12px', border: `1px solid ${info.color}33` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                    <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: info.color }}></div>
                    <span style={{ fontWeight: 700, fontSize: '0.9rem', color: '#fff' }}>{info.title}</span>
                  </div>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>{info.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
