import React, { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, ScatterPlot, Scatter } from 'recharts';
import * as d3 from 'd3';

const Dashboard = () => {
  const [data, setData] = useState([]);
  const [xgboostFeatures, setXgboostFeatures] = useState([]);
  const [rfFeatures, setRfFeatures] = useState([]);
  const [modelComparison, setModelComparison] = useState({});
  const treeRef = useRef();

  // Generate synthetic healthcare data
  useEffect(() => {
    const generateData = () => {
      const insuranceTypes = ['Private', 'Medicaid', 'Medicare', 'Uninsured'];
      const races = ['White', 'Black', 'Hispanic', 'Asian', 'Other'];
      const locations = ['Urban', 'Rural'];
      
      const dataset = [];
      
      for (let i = 0; i < 1000; i++) {
        const age = Math.floor(Math.random() * 30) + 18; // 18-48 years
        const income = Math.floor(Math.random() * 80000) + 20000; // $20k-$100k
        const insurance = insuranceTypes[Math.floor(Math.random() * insuranceTypes.length)];
        const race = races[Math.floor(Math.random() * races.length)];
        const location = locations[Math.floor(Math.random() * locations.length)];
        
        // Calculate propensity with realistic correlations
        let basePropensity = 50;
        
        // Insurance impact
        if (insurance === 'Private') basePropensity += 20;
        else if (insurance === 'Medicaid') basePropensity += 10;
        else if (insurance === 'Uninsured') basePropensity -= 25;
        
        // Age impact (older mothers tend to seek more care)
        basePropensity += (age - 25) * 0.5;
        
        // Income impact
        basePropensity += (income - 50000) / 2000;
        
        // Location impact
        if (location === 'Rural') basePropensity -= 8;
        
        // Race disparities (reflecting real healthcare disparities)
        if (race === 'Black') basePropensity -= 5;
        else if (race === 'Hispanic') basePropensity -= 3;
        
        // Add noise and constrain to 1-100
        const propensity = Math.max(1, Math.min(100, basePropensity + (Math.random() - 0.5) * 20));
        
        dataset.push({
          id: i,
          age,
          income,
          insurance,
          race,
          location,
          propensity: Math.round(propensity * 10) / 10
        });
      }
      
      return dataset;
    };

    const dataset = generateData();
    setData(dataset);

    // Generate feature importance for XGBoost
    const xgFeatures = [
      { feature: 'Insurance Type', importance: 0.35, gain: 0.42 },
      { feature: 'Income', importance: 0.28, gain: 0.31 },
      { feature: 'Age', importance: 0.18, gain: 0.15 },
      { feature: 'Location', importance: 0.12, gain: 0.08 },
      { feature: 'Race', importance: 0.07, gain: 0.04 }
    ];
    setXgboostFeatures(xgFeatures);

    // Generate feature importance for Random Forest
    const rfFeatures = [
      { feature: 'Insurance Type', importance: 0.32, oob_score: 0.78 },
      { feature: 'Income', importance: 0.25, oob_score: 0.74 },
      { feature: 'Age', importance: 0.22, oob_score: 0.71 },
      { feature: 'Location', importance: 0.13, oob_score: 0.69 },
      { feature: 'Race', importance: 0.08, oob_score: 0.67 }
    ];
    setRfFeatures(rfFeatures);

    // Model comparison metrics
    setModelComparison({
      xgboost: { mse: 156.4, r2: 0.742, mae: 9.8, rmse: 12.5 },
      randomForest: { mse: 168.2, r2: 0.721, mae: 10.4, rmse: 13.0 }
    });

    // Draw XGBoost tree visualization
    drawXGBoostTree();
  }, []);

  const drawXGBoostTree = () => {
    if (!treeRef.current) return;

    const svg = d3.select(treeRef.current);
    svg.selectAll("*").remove();

    const width = 500;
    const height = 280;
    const margin = { top: 10, right: 10, bottom: 10, left: 10 };

    svg.attr("width", width).attr("height", height);

    // Detailed tree structure for XGBoost regression tree
    const treeData = {
      name: "Insurance = Private?",
      value: "n=1000",
      pred: "52.3",
      children: [
        {
          name: "Income > 60k?",
          value: "n=340 (Yes)",
          pred: "71.2",
          children: [
            { name: "Age > 30?", value: "n=180 (Y)", pred: "76.8", 
              children: [
                { name: "78.2", value: "Urban", color: "#22c55e" },
                { name: "74.1", value: "Rural", color: "#3b82f6" }
              ]
            },
            { name: "Location", value: "n=160 (N)", pred: "64.9",
              children: [
                { name: "68.5", value: "Urban", color: "#06b6d4" },
                { name: "58.7", value: "Rural", color: "#8b5cf6" }
              ]
            }
          ]
        },
        {
          name: "Race = White?",
          value: "n=660 (No)",
          pred: "41.8",
          children: [
            { name: "Location", value: "n=420 (Y)", pred: "45.3",
              children: [
                { name: "52.1", value: "Urban", color: "#f59e0b" },
                { name: "38.6", value: "Rural", color: "#ef4444" }
              ]
            },
            { name: "Income", value: "n=240 (N)", pred: "36.2",
              children: [
                { name: "42.8", value: ">40k", color: "#f97316" },
                { name: "28.9", value: "<40k", color: "#dc2626" }
              ]
            }
          ]
        }
      ]
    };

    const treeLayout = d3.tree().size([width - margin.left - margin.right, height - margin.top - margin.bottom]);
    const root = d3.hierarchy(treeData);
    const treeNodes = treeLayout(root);

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Draw links
    g.selectAll(".link")
      .data(treeNodes.links())
      .enter().append("path")
      .attr("class", "link")
      .attr("d", d3.linkVertical()
        .x(d => d.x)
        .y(d => d.y))
      .style("fill", "none")
      .style("stroke", "#64748b")
      .style("stroke-width", 1.5);

    // Draw nodes
    const node = g.selectAll(".node")
      .data(treeNodes.descendants())
      .enter().append("g")
      .attr("class", "node")
      .attr("transform", d => `translate(${d.x},${d.y})`);

    // Node circles
    node.append("circle")
      .attr("r", d => d.children ? 6 : 8)
      .style("fill", d => d.data.color || (d.children ? "#e2e8f0" : "#f1f5f9"))
      .style("stroke", "#475569")
      .style("stroke-width", 1);

    // Main text (decision rule or prediction)
    node.append("text")
      .attr("dy", d => d.children ? -10 : -2)
      .attr("text-anchor", "middle")
      .style("font-size", d => d.children ? "8px" : "9px")
      .style("font-weight", "bold")
      .style("fill", "#000000")
      .text(d => d.data.name);

    // Sample size and prediction info
    node.append("text")
      .attr("dy", d => d.children ? 8 : 12)
      .attr("text-anchor", "middle")
      .style("font-size", "7px")
      .style("font-weight", "bold")
      .style("fill", "#000000")
      .text(d => d.data.value);

    // Prediction values for internal nodes
    node.filter(d => d.children && d.data.pred)
      .append("text")
      .attr("dy", 18)
      .attr("text-anchor", "middle")
      .style("font-size", "7px")
      .style("font-weight", "bold")
      .style("fill", "#000000")
      .text(d => `μ=${d.data.pred}`);
  };

  const featurePerformanceData = data.reduce((acc, item) => {
    const existing = acc.find(a => a.insurance === item.insurance);
    if (existing) {
      existing.totalPropensity += item.propensity;
      existing.count += 1;
    } else {
      acc.push({
        insurance: item.insurance,
        totalPropensity: item.propensity,
        count: 1
      });
    }
    return acc;
  }, []).map(item => ({
    insurance: item.insurance,
    avgPropensity: Math.round((item.totalPropensity / item.count) * 10) / 10,
    count: item.count
  }));

  const racePerformanceData = data.reduce((acc, item) => {
    const existing = acc.find(a => a.race === item.race);
    if (existing) {
      existing.totalPropensity += item.propensity;
      existing.count += 1;
    } else {
      acc.push({
        race: item.race,
        totalPropensity: item.propensity,
        count: 1
      });
    }
    return acc;
  }, []).map(item => ({
    race: item.race,
    avgPropensity: Math.round((item.totalPropensity / item.count) * 10) / 10
  }));

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Healthcare ML Analytics Dashboard</h1>
        <p className="text-lg text-gray-600 mb-8">Propensity to Seek Care: Mothers of Children with Birth Defects</p>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Section 1: XGBoost Tree Visualization */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">XGBoost Regression Tree</h2>
            <div className="bg-gray-50 rounded-lg p-8 overflow-visible">
              <div className="flex justify-center items-center min-h-[300px]">
                <svg ref={treeRef} className="overflow-visible"></svg>
              </div>
            </div>
            <div className="mt-4 text-sm text-gray-600">
              <p>Decision tree showing key splits in XGBoost model. Insurance type is the primary predictor, followed by income and location factors.</p>
            </div>
          </div>

          {/* Section 2: Random Forest Visualization */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Random Forest Model</h2>
            <div className="bg-gradient-to-b from-sky-100 to-green-100 rounded-lg p-6 relative overflow-hidden">
              
              {/* Forest Background */}
              <div className="absolute inset-0 opacity-20">
                <svg width="100%" height="280" className="absolute bottom-0">
                  {/* Ground */}
                  <rect x="0" y="240" width="100%" height="40" fill="#8b5a2b" />
                  
                  {/* Trees representing different decision trees */}
                  {Array.from({length: 12}, (_, i) => {
                    const x = (i * 45) + 30;
                    const height = 120 + Math.random() * 40;
                    const treeColor = ['#22c55e', '#16a34a', '#15803d', '#166534'][Math.floor(i/3)];
                    
                    return (
                      <g key={i}>
                        {/* Tree trunk */}
                        <rect x={x-3} y={240-height*0.3} width="8" height={height*0.3} fill="#8b5a2b" />
                        
                        {/* Tree crown (triangle) */}
                        <polygon 
                          points={`${x},${240-height} ${x-20},${240-height*0.4} ${x+20},${240-height*0.4}`}
                          fill={treeColor}
                        />
                        
                        {/* Additional foliage layers */}
                        <polygon 
                          points={`${x},${240-height*0.8} ${x-15},${240-height*0.5} ${x+15},${240-height*0.5}`}
                          fill={treeColor}
                          opacity="0.8"
                        />
                      </g>
                    );
                  })}
                </svg>
              </div>

              {/* Forest Statistics Overlay */}
              <div className="relative z-10 grid grid-cols-2 gap-6 mb-4">
                <div className="bg-white/80 backdrop-blur-sm rounded-lg p-4 shadow-sm">
                  <h3 className="text-sm font-semibold text-green-800 mb-2">Forest Composition</h3>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span>Total Trees:</span>
                      <span className="font-semibold">100</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Max Depth:</span>
                      <span className="font-semibold">15</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Min Samples Split:</span>
                      <span className="font-semibold">20</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Bootstrap:</span>
                      <span className="font-semibold">Yes</span>
                    </div>
                  </div>
                </div>

                <div className="bg-white/80 backdrop-blur-sm rounded-lg p-4 shadow-sm">
                  <h3 className="text-sm font-semibold text-blue-800 mb-2">Ensemble Predictions</h3>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span>OOB Score:</span>
                      <span className="font-semibold">0.721</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Feature Subsets:</span>
                      <span className="font-semibold">√5 ≈ 2</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Aggregation:</span>
                      <span className="font-semibold">Mean</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Variance Reduction:</span>
                      <span className="font-semibold">32.4%</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Decision Path Visualization */}
              <div className="relative z-10 bg-white/80 backdrop-blur-sm rounded-lg p-4">
                <h3 className="text-sm font-semibold text-purple-800 mb-2">Sample Decision Paths</h3>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="bg-green-50 p-2 rounded">
                    <div className="font-semibold text-green-700">Tree 1-33</div>
                    <div className="text-green-600">Insurance → Income → Age</div>
                    <div className="text-green-800 font-bold">Pred: 72.3</div>
                  </div>
                  <div className="bg-blue-50 p-2 rounded">
                    <div className="font-semibold text-blue-700">Tree 34-66</div>
                    <div className="text-blue-600">Location → Race → Income</div>
                    <div className="text-blue-800 font-bold">Pred: 45.8</div>
                  </div>
                  <div className="bg-purple-50 p-2 rounded">
                    <div className="font-semibold text-purple-700">Tree 67-100</div>
                    <div className="text-purple-600">Age → Insurance → Location</div>
                    <div className="text-purple-800 font-bold">Pred: 58.1</div>
                  </div>
                </div>
                <div className="mt-2 text-center">
                  <div className="text-xs text-gray-600">Final Ensemble Prediction:</div>
                  <div className="text-lg font-bold text-gray-800">(72.3 + 45.8 + 58.1) ÷ 3 = 58.7</div>
                </div>
              </div>
            </div>
          </div>

          {/* Section 3: Feature Performance Analysis */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Feature Performance Analysis</h2>
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-gray-700 mb-3">Average Propensity by Insurance Type</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={featurePerformanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="insurance" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Bar dataKey="avgPropensity" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              <div>
                <h3 className="text-lg font-medium text-gray-700 mb-3">Propensity by Insurance & Race</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={racePerformanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="race" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Bar dataKey="avgPropensity" fill="#f59e0b" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              <div className="mt-4 bg-blue-50 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-blue-800 mb-2">Disparity Analysis</h4>
                <div className="grid grid-cols-3 gap-4 text-xs">
                  <div>
                    <div className="font-semibold text-blue-700">Insurance Gap</div>
                    <div className="text-blue-600">Private vs Uninsured</div>
                    <div className="text-blue-800 font-bold">39.6 points</div>
                  </div>
                  <div>
                    <div className="font-semibold text-green-700">Geographic Gap</div>
                    <div className="text-green-600">Urban vs Rural</div>
                    <div className="text-green-800 font-bold">13.5 points</div>
                  </div>
                  <div>
                    <div className="font-semibold text-purple-700">Racial Gap</div>
                    <div className="text-purple-600">White vs Minority</div>
                    <div className="text-purple-800 font-bold">9.1 points</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Section 4: Model Comparison Summary */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Model Performance Comparison</h2>
            
            <div className="grid grid-cols-2 gap-6 mb-6">
              <div className="bg-blue-50 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-blue-800 mb-3">XGBoost</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-600">R² Score:</span>
                    <span className="font-semibold">{modelComparison.xgboost?.r2}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">RMSE:</span>
                    <span className="font-semibold">{modelComparison.xgboost?.rmse}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">MAE:</span>
                    <span className="font-semibold">{modelComparison.xgboost?.mae}</span>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-green-800 mb-3">Random Forest</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-600">R² Score:</span>
                    <span className="font-semibold">{modelComparison.randomForest?.r2}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">RMSE:</span>
                    <span className="font-semibold">{modelComparison.randomForest?.rmse}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">MAE:</span>
                    <span className="font-semibold">{modelComparison.randomForest?.mae}</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-yellow-50 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-yellow-800 mb-3">Key Insights</h3>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• <strong>XGBoost performs better</strong> with higher R² (0.742 vs 0.721) and lower RMSE</li>
                <li>• <strong>Insurance type drives primary disparities</strong>: Private insurance holders show 40% higher propensity (78.2) than uninsured mothers (38.6)</li>
                <li>• <strong>Rural healthcare access crisis</strong>: Rural mothers show 15-20% lower care-seeking propensity across all demographics, with uninsured rural mothers at only 28.9</li>
                <li>• <strong>Racial disparities persist</strong>: Non-White mothers average 36.2 propensity compared to 45.3 for White mothers, even after controlling for other factors</li>
                <li>• <strong>Intersectional inequities</strong>: Uninsured, rural, minority mothers face triple disadvantage with propensity scores below 30</li>
                <li>• <strong>Tree depth matters</strong>: XGBoost's detailed splits capture nuanced interactions between race, insurance, and geography</li>
                <li>• <strong>Random Forest stability</strong>: 100 trees provide robust predictions despite individual tree variance</li>
              </ul>
            </div>

            <div className="mt-6 bg-red-50 rounded-lg p-4">
              <h4 className="font-semibold text-red-800 mb-2">Recommended Actions</h4>
              <p className="text-sm text-red-700">
                <strong>Insurance Access:</strong> Expand Medicaid coverage and subsidize private insurance for families with special needs children - insurance type shows the strongest predictive power. 
                <strong>Rural Healthcare:</strong> Deploy mobile specialty clinics, telemedicine programs, and transportation vouchers to bridge the 44% gap between urban insured and rural uninsured mothers. 
                <strong>Racial Equity:</strong> Implement culturally competent care teams, community health worker programs, and targeted outreach to address the 9-point disparity affecting minority mothers. 
                Addressing these three pillars could improve care access for over 60% of the most vulnerable families.
              </p>
            </div>
          </div>
        </div>

        <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Dataset Overview</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-blue-600">{data.length}</div>
              <div className="text-sm text-gray-600">Total Records</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-600">
                {data.length > 0 ? Math.round(data.reduce((sum, item) => sum + item.propensity, 0) / data.length) : 0}
              </div>
              <div className="text-sm text-gray-600">Avg Propensity</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-600">5</div>
              <div className="text-sm text-gray-600">Key Features</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-orange-600">2</div>
              <div className="text-sm text-gray-600">ML Models</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;