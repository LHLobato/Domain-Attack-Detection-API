import { HashRouter, Routes, Route } from 'react-router-dom';


import { NavBar } from './components/Navbar';
import { HomePage } from './components/HomePage'; 
import { DocPage } from './components/DocPage';

function App() {

  
  return (
    <HashRouter>
      
      <NavBar /> 

      <Routes>

        <Route path="/" element={<HomePage />} />


        <Route path="/docs" element={<DocPage />} />

      </Routes>
      
    </HashRouter>
  );
}

export default App;