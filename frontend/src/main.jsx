import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import VideoFeed from '../pages/VideoFeed.jsx'
import './index.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <VideoFeed />
  </StrictMode>,
)
