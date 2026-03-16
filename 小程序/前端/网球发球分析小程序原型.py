import React, { useState, useEffect } from 'react';
import { 
  Camera, 
  Upload, 
  Activity, 
  ChevronRight, 
  User, 
  BarChart2, 
  Play,
  Pause,
  CheckCircle, 
  AlertTriangle, 
  TrendingUp,
  Info,
  Settings,
  Award,
  Zap,
  Share,
  X,
  Copy,
  Download,
  MessageCircle,
  Moon,
  Bell,
  Volume2
} from 'lucide-react';

// --- 模拟数据 ---
const MOCK_ANALYSIS_RESULT = {
  score: 82,
  date: "2023-10-24",
  type: "直线平击发球",
  summary: "发力链传导良好，但在奖杯姿势阶段存在微小瑕疵。",
  metrics: [
    { label: "膝盖弯曲", value: "115°", status: "optimal", ideal: "110°-120°" },
    { label: "髋肩分离", value: "28°", status: "warning", ideal: ">30°" },
    { label: "击球点高度", value: "2.85m", status: "optimal", ideal: "Max Reach" },
    { label: "拍头速度", value: "112 mph", status: "good", ideal: "Target >100" }
  ],
  drill: {
    title: "药球过顶投掷",
    difficulty: "中级",
    duration: "10 分钟",
    desc: "针对髋肩分离不足，建议练习跪姿药球过顶投掷。这个动作能强制隔离下肢，迫使你利用胸椎的延展和核心的扭转来产生力量。",
    steps: [
        "双膝跪地，保持核心收紧。",
        "双手持药球于脑后，手肘指向天空。",
        "向后反弓身体，感受胸肌拉伸。",
        "爆发性向前投掷药球，注意不要过度弯腰。"
    ]
  }
};

const MOCK_HISTORY = [
  { id: 1, date: "10月24日", type: "平击发球", score: 82, trend: 'up' },
  { id: 2, date: "10月22日", type: "侧旋发球", score: 78, trend: 'down' },
  { id: 3, date: "10月18日", type: "上旋发球", score: 85, trend: 'up' },
  { id: 4, date: "10月15日", type: "平击发球", score: 76, trend: 'flat' },
];

const App = () => {
  const [activeTab, setActiveTab] = useState('home');
  const [appState, setAppState] = useState('idle'); 
  const [progress, setProgress] = useState(0);
  const [selectedHistoryId, setSelectedHistoryId] = useState(null);

  // 新增交互状态
  const [isPlaying, setIsPlaying] = useState(false);
  const [showShareSheet, setShowShareSheet] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showDrillDetail, setShowDrillDetail] = useState(false);
  const [darkMode, setDarkMode] = useState(false);

  // 模拟上传
  const handleUpload = () => {
    setAppState('uploading');
    setTimeout(() => {
      setAppState('analyzing');
      let p = 0;
      const interval = setInterval(() => {
        p += 5;
        setProgress(p);
        if (p >= 100) {
          clearInterval(interval);
          setAppState('result');
        }
      }, 50);
    }, 1000);
  };

  const resetAnalysis = () => {
    setAppState('idle');
    setProgress(0);
    setSelectedHistoryId(null);
    setIsPlaying(false);
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
  };

  // --- 通用 UI 组件 ---

  const IOSGroup = ({ children, title }) => (
    <div className="mb-6">
      {title && <div className="px-4 mb-2 text-xs font-medium text-gray-500 uppercase tracking-wide">{title}</div>}
      <div className="bg-white rounded-[10px] overflow-hidden mx-4 shadow-[0_1px_2px_rgba(0,0,0,0.02)]">
        {children}
      </div>
    </div>
  );

  const IOSListItem = ({ icon: Icon, label, value, onClick, isLast, color = "bg-blue-500", toggle }) => (
    <div 
      onClick={onClick}
      className={`flex items-center justify-between p-4 active:bg-gray-50 transition-colors cursor-pointer ${!isLast ? 'border-b border-gray-100' : ''}`}
    >
      <div className="flex items-center space-x-3">
        {Icon && (
          <div className={`w-7 h-7 rounded-[7px] flex items-center justify-center text-white ${color}`}>
            <Icon size={16} />
          </div>
        )}
        <span className="text-[17px] text-gray-900 font-medium">{label}</span>
      </div>
      <div className="flex items-center space-x-2">
        {value && <span className="text-[17px] text-gray-500">{value}</span>}
        {toggle !== undefined ? (
           <div className={`w-[51px] h-[31px] rounded-full p-0.5 transition-colors ${toggle ? 'bg-green-500' : 'bg-gray-200'}`}>
              <div className={`w-[27px] h-[27px] bg-white rounded-full shadow-sm transform transition-transform ${toggle ? 'translate-x-[20px]' : 'translate-x-0'}`} />
           </div>
        ) : (
           <ChevronRight size={18} className="text-gray-300" />
        )}
      </div>
    </div>
  );

  // --- 模态窗组件 ---

  const ShareSheet = () => (
    <div className={`fixed inset-0 z-[60] flex flex-col justify-end transition-opacity duration-300 ${showShareSheet ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}>
       <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={() => setShowShareSheet(false)}></div>
       <div className={`bg-[#F2F2F7] rounded-t-[20px] p-4 transform transition-transform duration-300 ${showShareSheet ? 'translate-y-0' : 'translate-y-full'} z-10 safe-pb`}>
          <div className="flex justify-between items-center mb-4 px-2">
             <h3 className="text-lg font-semibold text-gray-900">分享分析报告</h3>
             <button onClick={() => setShowShareSheet(false)} className="bg-gray-200 p-1 rounded-full"><X size={16}/></button>
          </div>
          
          <div className="bg-white rounded-xl p-4 mb-4 flex items-center space-x-4 shadow-sm">
             <div className="w-12 h-12 bg-black rounded-lg flex items-center justify-center text-white font-bold text-xs">IMG</div>
             <div>
                <div className="font-semibold text-sm">发球分析_1024.pdf</div>
                <div className="text-xs text-gray-400">2.4 MB • PDF 文档</div>
             </div>
          </div>

          <div className="bg-white rounded-xl overflow-hidden mb-4 shadow-sm">
             <button className="w-full p-4 flex items-center space-x-3 hover:bg-gray-50 border-b border-gray-100">
                <MessageCircle size={20} className="text-green-500"/>
                <span className="text-[17px]">发送给教练</span>
             </button>
             <button className="w-full p-4 flex items-center space-x-3 hover:bg-gray-50 border-b border-gray-100">
                <Copy size={20} className="text-gray-500"/>
                <span className="text-[17px]">复制链接</span>
             </button>
             <button className="w-full p-4 flex items-center space-x-3 hover:bg-gray-50">
                <Download size={20} className="text-blue-500"/>
                <span className="text-[17px]">保存图片</span>
             </button>
          </div>
       </div>
    </div>
  );

  const SettingsModal = () => (
     <div className={`fixed inset-0 z-[60] bg-[#F2F2F7] transform transition-transform duration-300 ${showSettings ? 'translate-y-0' : 'translate-y-full'}`}>
        <div className="bg-white px-4 py-3 border-b border-gray-200 flex justify-between items-center sticky top-0 z-10 pt-12">
           <div className="w-10"></div>
           <h2 className="font-semibold text-[17px]">设置</h2>
           <button onClick={() => setShowSettings(false)} className="text-blue-600 font-semibold text-[17px] w-10 text-right">完成</button>
        </div>
        <div className="pt-6">
           <IOSGroup title="通用">
              <IOSListItem icon={Moon} label="深色模式" color="bg-gray-800" toggle={darkMode} onClick={() => setDarkMode(!darkMode)} />
              <IOSListItem icon={Bell} label="推送通知" color="bg-red-500" toggle={true} />
              <IOSListItem icon={Volume2} label="音效" color="bg-pink-500" toggle={false} isLast />
           </IOSGroup>
           <IOSGroup title="账号">
              <IOSListItem label="修改密码" isLast />
           </IOSGroup>
           <div className="px-8 mt-8">
              <p className="text-center text-xs text-gray-400">BioServe Pro v1.0.2 (Build 2024)</p>
           </div>
        </div>
     </div>
  );

  const DrillModal = () => (
    <div className={`fixed inset-0 z-[60] bg-white transform transition-transform duration-300 ${showDrillDetail ? 'translate-x-0' : 'translate-x-full'}`}>
       <div className="px-4 py-3 border-b border-gray-100 flex items-center sticky top-0 bg-white z-10 pt-12">
           <button onClick={() => setShowDrillDetail(false)} className="flex items-center text-blue-600 text-[17px] -ml-2">
             <ChevronRight className="rotate-180 w-6 h-6 mr-[-4px]" /> 返回
           </button>
           <h2 className="font-semibold text-[17px] mx-auto pr-12">训练详情</h2>
       </div>
       <div className="overflow-y-auto h-full pb-32">
          {/* 视频演示区 */}
          <div className="w-full h-56 bg-gray-100 flex items-center justify-center relative">
             <Play className="fill-gray-900 text-gray-900 w-12 h-12 opacity-50" />
             <div className="absolute bottom-2 right-2 bg-black/50 text-white text-xs px-2 py-1 rounded">演示视频</div>
          </div>
          
          <div className="p-5">
             <div className="flex space-x-2 mb-4">
               <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs font-medium">{MOCK_ANALYSIS_RESULT.drill.difficulty}</span>
               <span className="bg-gray-100 text-gray-600 px-2 py-1 rounded text-xs font-medium">{MOCK_ANALYSIS_RESULT.drill.duration}</span>
             </div>
             
             <h1 className="text-2xl font-bold text-gray-900 mb-2">{MOCK_ANALYSIS_RESULT.drill.title}</h1>
             <p className="text-gray-600 leading-relaxed mb-8">{MOCK_ANALYSIS_RESULT.drill.desc}</p>
             
             <h3 className="font-bold text-lg mb-4">训练步骤</h3>
             <div className="space-y-6 relative">
                <div className="absolute top-2 left-[11px] h-full w-[2px] bg-gray-100 z-0"></div>
                {MOCK_ANALYSIS_RESULT.drill.steps.map((step, i) => (
                   <div key={i} className="flex space-x-4 relative z-10">
                      <div className="w-6 h-6 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">
                         {i + 1}
                      </div>
                      <p className="text-gray-800 pt-0.5">{step}</p>
                   </div>
                ))}
             </div>
          </div>
          <div className="p-5">
             <button onClick={() => setShowDrillDetail(false)} className="w-full bg-blue-600 text-white py-3.5 rounded-xl font-bold active:scale-95 transition-transform">
                开始训练
             </button>
          </div>
       </div>
    </div>
  );

  const renderHomeContent = () => {
    // 状态页面：分析中 (保持不变)
    if (appState === 'uploading' || appState === 'analyzing') {
      return (
        <div className="h-full flex flex-col items-center justify-center p-8 bg-[#F2F2F7] z-50 absolute top-0 left-0 w-full animate-fade-in">
          <div className="bg-white/80 backdrop-blur-xl p-8 rounded-2xl shadow-lg flex flex-col items-center w-64">
             <div className="relative w-16 h-16 mb-4">
                <svg className="animate-spin w-full h-full text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-blue-600">{progress}%</div>
             </div>
             <h3 className="text-gray-900 font-semibold mb-1">正在分析</h3>
             <p className="text-gray-500 text-xs text-center">
                {appState === 'uploading' ? '正在处理视频...' : 'AI 正在计算骨骼角度...'}
             </p>
          </div>
        </div>
      );
    }

    // 状态页面：分析结果
    if (appState === 'result' || selectedHistoryId) {
      const data = selectedHistoryId ? MOCK_HISTORY.find(h=>h.id===selectedHistoryId) : MOCK_ANALYSIS_RESULT;
      const score = data?.score || 82;

      return (
        <div className="pb-24 animate-fade-in bg-[#F2F2F7] min-h-full">
          {/* 导航栏 */}
          <div className="sticky top-0 z-30 bg-white/80 backdrop-blur-md border-b border-gray-200/50 pt-12 pb-3 px-4 flex items-center justify-between">
            <button 
              onClick={resetAnalysis} 
              className="flex items-center text-blue-600 text-[17px]"
            >
              <ChevronRight className="rotate-180 w-6 h-6 mr-[-4px]" />
              返回
            </button>
            <span className="font-semibold text-[17px]">分析报告</span>
            <button onClick={() => setShowShareSheet(true)} className="text-blue-600">
               <Share size={20} />
            </button>
          </div>

          <div className="p-4 space-y-6">
            {/* 交互式视频播放器 */}
            <div className="bg-black rounded-xl overflow-hidden shadow-sm relative h-64 group">
               {/* 模拟视频画面 */}
               <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
                  <svg className="absolute inset-0 w-full h-full opacity-30" viewBox="0 0 100 100">
                      <line x1="50" y1="20" x2="50" y2="50" stroke="lime" strokeWidth="1" />
                      <line x1="50" y1="50" x2="30" y2="80" stroke="lime" strokeWidth="1" />
                      <line x1="50" y1="50" x2="70" y2="80" stroke="lime" strokeWidth="1" />
                  </svg>
               </div>
               
               {/* 播放控制层 */}
               <div className="absolute inset-0 flex items-center justify-center z-10" onClick={() => setIsPlaying(!isPlaying)}>
                  {!isPlaying && (
                     <div className="w-16 h-16 bg-black/40 rounded-full flex items-center justify-center backdrop-blur-sm shadow-lg transform transition-transform active:scale-95">
                        <Play className="text-white ml-1 fill-white" size={32} />
                     </div>
                  )}
               </div>

               {/* 底部控制栏 */}
               <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/80 to-transparent z-20">
                  <div className="flex items-center space-x-3 text-white text-xs font-medium">
                     <button onClick={() => setIsPlaying(!isPlaying)}>
                        {isPlaying ? <Pause size={16} className="fill-white"/> : <Play size={16} className="fill-white"/>}
                     </button>
                     <div className="flex-1 h-1 bg-white/30 rounded-full overflow-hidden relative">
                        <div className={`absolute top-0 left-0 h-full bg-blue-500 rounded-full ${isPlaying ? 'w-2/3 transition-all duration-[3s] ease-linear' : 'w-1/3'}`}></div>
                     </div>
                     <span>00:04</span>
                  </div>
               </div>
            </div>

            {/* 分数概览 (保持不变) */}
            <div className="bg-white rounded-xl p-5 shadow-[0_1px_2px_rgba(0,0,0,0.02)] flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold tracking-tight text-gray-900">{score} <span className="text-lg font-medium text-gray-400">分</span></h1>
                <p className="text-gray-500 text-sm mt-1">{MOCK_ANALYSIS_RESULT.summary}</p>
              </div>
              <div className="w-16 h-16 relative">
                 <svg viewBox="0 0 36 36" className="w-full h-full rotate-[-90deg]">
                    <path className="text-gray-100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeWidth="3" />
                    <path className={score >= 80 ? "text-green-500" : "text-orange-500"} strokeDasharray={`${score}, 100`} d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                 </svg>
              </div>
            </div>

            {/* 关键指标 */}
            <div>
              <h3 className="text-lg font-bold text-gray-900 mb-3 px-1">详细数据</h3>
              <div className="grid grid-cols-2 gap-3">
                {MOCK_ANALYSIS_RESULT.metrics.map((metric, index) => (
                  <div key={index} className="bg-white p-4 rounded-xl shadow-[0_1px_2px_rgba(0,0,0,0.02)] flex flex-col justify-between h-28">
                    <div className="flex justify-between items-start">
                      <div className="font-semibold text-[11px] text-gray-400 uppercase tracking-wide flex items-center gap-1">
                        {metric.status === 'optimal' && <div className="w-1.5 h-1.5 rounded-full bg-green-500"></div>}
                        {metric.status === 'warning' && <div className="w-1.5 h-1.5 rounded-full bg-orange-500"></div>}
                        {metric.label.split(' ')[0]}
                      </div>
                    </div>
                    <div>
                      <div className="text-xl font-bold text-gray-900">{metric.value}</div>
                      <div className="text-xs text-gray-400 mt-1">目标: {metric.ideal}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Coach's Corner (可点击) */}
            <div onClick={() => setShowDrillDetail(true)} className="bg-white/60 backdrop-blur-md border border-white/60 rounded-2xl p-4 shadow-sm relative overflow-hidden active:scale-[0.98] transition-transform cursor-pointer">
               <div className="absolute top-0 left-0 w-1.5 h-full bg-blue-500"></div>
               <div className="pl-3 flex justify-between items-center">
                 <div>
                    <div className="flex items-center space-x-2 mb-1">
                        <div className="bg-blue-100 p-1 rounded text-blue-600"><Zap size={12}/></div>
                        <span className="text-xs font-bold text-gray-500 uppercase">Coach's Suggestion</span>
                    </div>
                    <h4 className="font-semibold text-gray-900 mb-1">{MOCK_ANALYSIS_RESULT.drill.title}</h4>
                    <p className="text-sm text-gray-600 leading-relaxed truncate w-60">{MOCK_ANALYSIS_RESULT.drill.desc}</p>
                 </div>
                 <ChevronRight className="text-gray-300" />
               </div>
            </div>

          </div>
        </div>
      );
    }

    // Default Home View
    return (
      <div className="animate-fade-in bg-[#F2F2F7] min-h-full pb-24">
        {/* Apple Style Large Title Header */}
        <div className="pt-14 pb-2 px-5 bg-[#F2F2F7]">
          <div className="text-xs font-semibold text-gray-500 uppercase mb-1">{new Date().toLocaleDateString('zh-CN', {weekday:'long', month:'long', day:'numeric'})}</div>
          <div className="flex justify-between items-end">
            <h1 className="text-[34px] font-bold text-gray-900 leading-tight">概要</h1>
            <div className="w-9 h-9 bg-gray-200 rounded-full overflow-hidden border border-white/50 shadow-sm">
               <div className="w-full h-full flex items-center justify-center bg-gray-300 text-gray-500 font-bold text-xs">JP</div>
            </div>
          </div>
        </div>

        <div className="px-5 space-y-6 mt-4">
          
          {/* Main Action Card (Updated Background) */}
          <div className="relative h-96 rounded-[20px] overflow-hidden shadow-md group cursor-pointer" onClick={handleUpload}>
            {/* 抽象网球背景图 - SVG 实现 */}
            <div className="absolute inset-0 bg-[#1c1c1e]">
               <svg className="w-full h-full object-cover" viewBox="0 0 400 500" preserveAspectRatio="xMidYMid slice">
                  <defs>
                     <linearGradient id="cardGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#2c3e50" />
                        <stop offset="100%" stopColor="#000000" />
                     </linearGradient>
                  </defs>
                  <rect width="100%" height="100%" fill="url(#cardGrad)" />
                  
                  {/* 网球场透视线 */}
                  <g opacity="0.3">
                    <path d="M-50 500 L450 500" stroke="white" strokeWidth="2" />
                    <path d="M200 500 L200 100" stroke="white" strokeWidth="2" strokeDasharray="10 5"/>
                    <path d="M-100 500 L150 200" stroke="white" strokeWidth="1" />
                    <path d="M500 500 L250 200" stroke="white" strokeWidth="1" />
                    <circle cx="200" cy="200" r="100" stroke="white" strokeWidth="2" fill="none" opacity="0.5"/>
                  </g>

                  {/* 悬浮的网球 (艺术化) */}
                  <g transform="translate(260, 120)">
                     <circle cx="0" cy="0" r="50" fill="#dfff00" />
                     {/* 网球缝线 */}
                     <path d="M-35 -35 Q 0 0 35 -35" stroke="#c0dc00" strokeWidth="4" fill="none" />
                     <path d="M-35 35 Q 0 0 35 35" stroke="#c0dc00" strokeWidth="4" fill="none" />
                  </g>
                  
                  {/* 光影叠加 */}
                  <circle cx="400" cy="0" r="200" fill="url(#glow)" opacity="0.4" />
                  <defs>
                    <radialGradient id="glow">
                      <stop offset="0%" stopColor="#4facfe" />
                      <stop offset="100%" stopColor="transparent" />
                    </radialGradient>
                  </defs>
               </svg>
            </div>
            
            <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-transparent to-black/10"></div>
            
            <div className="absolute bottom-0 left-0 p-6 w-full">
              <div className="text-green-400 font-semibold text-xs uppercase tracking-wider mb-2">New Session</div>
              <h2 className="text-3xl font-bold text-white mb-2">发球分析</h2>
              <p className="text-gray-300 text-sm mb-6 line-clamp-2">使用 AI 技术，从生物力学角度拆解你的发球动作，提升效率，预防伤病。</p>
              
              <button className="bg-white text-black w-full py-3.5 rounded-full font-bold text-[17px] active:scale-95 transition-transform flex items-center justify-center space-x-2 shadow-lg">
                <Camera className="w-5 h-5" />
                <span>开始拍摄</span>
              </button>
            </div>
          </div>

          {/* Quick Actions Grid */}
          <div className="grid grid-cols-2 gap-4">
             <div onClick={handleUpload} className="bg-white rounded-2xl p-4 h-32 flex flex-col justify-between shadow-[0_1px_2px_rgba(0,0,0,0.02)] active:scale-95 transition-transform">
                <div className="w-8 h-8 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center">
                   <Upload size={18} />
                </div>
                <div>
                   <span className="block font-semibold text-gray-900">导入视频</span>
                   <span className="text-xs text-gray-400">从相册选择</span>
                </div>
             </div>
             
             <div className="bg-white rounded-2xl p-4 h-32 flex flex-col justify-between shadow-[0_1px_2px_rgba(0,0,0,0.02)]">
                <div className="flex justify-between items-start">
                   <div className="w-8 h-8 rounded-full bg-green-50 text-green-600 flex items-center justify-center">
                      <BarChart2 size={18} />
                   </div>
                   <span className="text-2xl font-bold text-gray-900">82</span>
                </div>
                <div>
                   <span className="block font-semibold text-gray-900">本周均分</span>
                   <span className="text-xs text-green-600 flex items-center">
                      <TrendingUp size={10} className="mr-1"/> +2.4
                   </span>
                </div>
             </div>
          </div>

          {/* Tip Card */}
          <div className="bg-white/80 backdrop-blur-xl p-4 rounded-2xl shadow-[0_2px_8px_rgba(0,0,0,0.04)] flex items-start space-x-3">
             <div className="bg-orange-500 rounded-lg p-1.5 shrink-0 mt-0.5">
               <Award className="text-white w-5 h-5" />
             </div>
             <div>
                <h3 className="font-semibold text-gray-900 text-sm">每日技巧</h3>
                <p className="text-gray-500 text-[13px] mt-1 leading-snug">
                   保持“奖杯姿势”时，尝试将左肩稍微抬高，这有助于增加躯干的侧向弯曲。
                </p>
             </div>
          </div>
        </div>
      </div>
    );
  };

  const renderHistoryContent = () => (
    <div className="animate-fade-in bg-[#F2F2F7] min-h-full pb-24 pt-14">
      <div className="px-5 mb-2">
         <h1 className="text-[34px] font-bold text-gray-900">历史记录</h1>
         <div className="bg-gray-200/80 p-0.5 rounded-lg flex mt-4 mb-6">
            {['全部', '平击', '旋转'].map((filter, i) => (
               <button key={i} className={`flex-1 text-[13px] font-medium py-1.5 rounded-[6px] transition-all ${i===0 ? 'bg-white text-black shadow-sm' : 'text-gray-500'}`}>
                  {filter}
               </button>
            ))}
         </div>
      </div>

      <IOSGroup title="本周">
        {MOCK_HISTORY.slice(0, 2).map((item, i) => (
           <div 
            key={item.id} 
            onClick={() => {
              setSelectedHistoryId(item.id);
              setActiveTab('home'); 
              setAppState('result'); 
            }}
            className={`flex justify-between items-center p-4 bg-white active:bg-gray-50 transition-colors cursor-pointer ${i !== 1 ? 'border-b border-gray-100 ml-4' : 'mx-0 px-4'}`}
           >
              <div className="flex items-center space-x-3">
                 <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold ${item.score >= 80 ? 'bg-green-100 text-green-700' : 'bg-orange-100 text-orange-600'}`}>
                    {item.score}
                 </div>
                 <div>
                    <div className="font-semibold text-gray-900 text-[15px]">{item.type}</div>
                    <div className="text-xs text-gray-400">{item.date}</div>
                 </div>
              </div>
              <ChevronRight size={16} className="text-gray-300" />
           </div>
        ))}
      </IOSGroup>

      <IOSGroup title="上周">
        {MOCK_HISTORY.slice(2).map((item, i) => (
           <div 
            key={item.id} 
            className={`flex justify-between items-center p-4 bg-white active:bg-gray-50 transition-colors cursor-pointer ${i !== MOCK_HISTORY.slice(2).length - 1 ? 'border-b border-gray-100 ml-4' : 'mx-0 px-4'}`}
           >
              <div className="flex items-center space-x-3">
                 <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold ${item.score >= 80 ? 'bg-green-100 text-green-700' : 'bg-orange-100 text-orange-600'}`}>
                    {item.score}
                 </div>
                 <div>
                    <div className="font-semibold text-gray-900 text-[15px]">{item.type}</div>
                    <div className="text-xs text-gray-400">{item.date}</div>
                 </div>
              </div>
              <ChevronRight size={16} className="text-gray-300" />
           </div>
        ))}
      </IOSGroup>
    </div>
  );

  const renderProfileContent = () => (
    <div className="animate-fade-in bg-[#F2F2F7] min-h-full pb-24 pt-14">
      <div className="px-5 mb-6 text-center">
         <div className="w-24 h-24 mx-auto bg-gradient-to-b from-gray-200 to-gray-300 rounded-full mb-3 border-4 border-white shadow-sm flex items-center justify-center text-2xl font-bold text-gray-500">
            JP
         </div>
         <h1 className="text-2xl font-bold text-gray-900">Jason Player</h1>
         <p className="text-gray-500 text-sm">jason.player@icloud.com</p>
      </div>

      <IOSGroup>
         <IOSListItem icon={Activity} label="生涯统计" bg="bg-blue-500" />
         <IOSListItem icon={Award} label="我的成就" bg="bg-yellow-500" />
         <IOSListItem icon={Zap} label="训练计划" bg="bg-green-500" isLast />
      </IOSGroup>

      <IOSGroup>
         <IOSListItem icon={Settings} label="设置" bg="bg-gray-500" onClick={() => setShowSettings(true)} />
         <IOSListItem icon={Info} label="关于" bg="bg-gray-500" isLast />
      </IOSGroup>

      <div className="px-4">
         <button className="w-full py-3 text-[17px] text-red-500 bg-white rounded-[10px] font-medium active:bg-gray-50 transition-colors shadow-sm">
            退出登录
         </button>
      </div>
    </div>
  );

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-200 font-sans text-gray-900 selection:bg-blue-100">
      <div className="w-full max-w-md h-[850px] bg-[#F2F2F7] shadow-2xl overflow-hidden relative flex flex-col border-x border-gray-300/50">
        
        <div className="absolute top-0 w-full h-12 z-40 flex justify-between items-center px-6 pt-2 pointer-events-none">
           <span className="text-sm font-semibold text-gray-900">9:41</span>
           <div className="flex space-x-1.5">
             <div className="w-4 h-2.5 bg-gray-900 rounded-[1px]"></div>
             <div className="w-0.5 h-2.5 bg-gray-900/30 rounded-[1px]"></div>
           </div>
        </div>

        {/* 模态窗挂载点 */}
        <ShareSheet />
        <SettingsModal />
        <DrillModal />

        <div className="flex-1 overflow-y-auto no-scrollbar relative">
          {activeTab === 'home' && renderHomeContent()}
          {activeTab === 'history' && renderHistoryContent()}
          {activeTab === 'profile' && renderProfileContent()}
        </div>

        <div className="bg-white/85 backdrop-blur-xl border-t border-gray-200/50 h-[83px] flex justify-around items-start pt-3 px-2 absolute bottom-0 w-full z-40">
          <button 
            onClick={() => handleTabChange('home')} 
            className={`flex flex-col items-center justify-center w-16 space-y-1 transition-colors ${activeTab === 'home' ? 'text-blue-600' : 'text-gray-400'}`}
          >
            <Activity size={26} strokeWidth={activeTab === 'home' ? 2.5 : 2} />
            <span className="text-[10px] font-medium">分析</span>
          </button>
          
          <button 
            onClick={() => handleTabChange('history')} 
            className={`flex flex-col items-center justify-center w-16 space-y-1 transition-colors ${activeTab === 'history' ? 'text-blue-600' : 'text-gray-400'}`}
          >
            <BarChart2 size={26} strokeWidth={activeTab === 'history' ? 2.5 : 2} />
            <span className="text-[10px] font-medium">记录</span>
          </button>

          <button 
            onClick={() => handleTabChange('profile')} 
            className={`flex flex-col items-center justify-center w-16 space-y-1 transition-colors ${activeTab === 'profile' ? 'text-blue-600' : 'text-gray-400'}`}
          >
            <User size={26} strokeWidth={activeTab === 'profile' ? 2.5 : 2} />
            <span className="text-[10px] font-medium">我的</span>
          </button>
        </div>

      </div>
    </div>
  );
};

export default App;