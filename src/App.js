import React from 'react';
import PlayerChart from './PlayerChart';
import StatPredictions from './StatPredictions';

function App() {
  return (
    <div>
      <h1>NBA Player Stats (Last 10 Games)</h1>
      <PlayerChart />
      <StatPredictions />
    </div>
  );
}

export default App;