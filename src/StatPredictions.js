import React, { useState } from 'react';
import axios from 'axios';

const StatPredictions = () => {
  const [player, setPlayer] = useState('');
  const [opponent, setOpponent] = useState('');
  const [isHome, setIsHome] = useState(true);
  const [daysRest, setDaysRest] = useState(1);
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchPredictions = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.get('http://localhost:5000/api/predict-all', {
        params: {
          player,
          opponent,
          is_home: isHome,
          days_rest: daysRest
        }
      });
      setPredictions(response.data.predictions);
    } catch (err) {
      console.error(err);
      setError('Error fetching predictions.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '40px auto', padding: '20px', border: '1px solid #ccc' }}>
      <h2>Predict Next Game Stats</h2>
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
        <input
          type="text"
          value={player}
          onChange={(e) => setPlayer(e.target.value)}
          placeholder="Player Name"
          style={{ flex: 1, padding: '10px' }}
        />
        <input
          type="text"
          value={opponent}
          onChange={(e) => setOpponent(e.target.value)}
          placeholder="Opponent (e.g., PHX)"
          style={{ flex: 1, padding: '10px' }}
        />
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
        <label>
          Home Game:&nbsp;
          <select
            value={isHome ? 'true' : 'false'}
            onChange={(e) => setIsHome(e.target.value === 'true')}
            style={{ padding: '10px' }}
          >
            <option value="true">Yes</option>
            <option value="false">No</option>
          </select>
        </label>
        <label>
          Days Rest:&nbsp;
          <input
            type="number"
            value={daysRest}
            onChange={(e) => setDaysRest(parseInt(e.target.value))}
            style={{ padding: '10px', width: '80px' }}
          />
        </label>
      </div>
      <button onClick={fetchPredictions} style={{ padding: '10px 20px', fontSize: '16px' }}>
        Get Predictions
      </button>
      {loading && <p>Loading predictions...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {predictions && (
        <div style={{ marginTop: '20px' }}>
          <h3>Predictions for your next game:</h3>
          <ul>
            <li>
              <strong>Points:</strong> {predictions.PTS.prediction} ± {predictions.PTS.margin_error}
            </li>
            <li>
              <strong>Assists:</strong> {predictions.AST.prediction} ± {predictions.AST.margin_error}
            </li>
            <li>
              <strong>Rebounds:</strong> {predictions.REB.prediction} ± {predictions.REB.margin_error}
            </li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default StatPredictions;