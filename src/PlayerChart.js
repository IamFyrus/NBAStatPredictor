// src/PlayerChart.js
import React, { useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const PlayerChart = () => {
  const [player, setPlayer] = useState('');
  const [stat, setStat] = useState('PTS');
  const [data, setData] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchStats = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.get('http://localhost:5000/api/player-stats', {
        params: { player, stat }
      });
      setData(response.data);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
      setError('Failed to fetch stats. Please check player name or try again later.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '40px' }}>
      <h2>NBA Player Stats</h2>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
        <input
          style={{ flex: 1, padding: '10px', fontSize: '16px' }}
          value={player}
          onChange={(e) => setPlayer(e.target.value)}
          placeholder="Enter player name"
        />
        <select style={{ padding: '10px', fontSize: '16px' }} value={stat} onChange={(e) => setStat(e.target.value)}>
          <option value="PTS">Points</option>
          <option value="AST">Assists</option>
          <option value="REB">Rebounds</option>
        </select>
        <button style={{ padding: '10px 20px', fontSize: '16px' }} onClick={fetchStats}>
          Get Stats
        </button>
      </div>
      {loading && <p>Loading stats...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {data.length > 0 && (
        <ResponsiveContainer width={600} height={300}>
          <LineChart data={data}>
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <CartesianGrid stroke="#ccc" />
            <Legend />
            <Line type="monotone" dataKey="value" stroke="#8884d8" />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};

export default PlayerChart;
