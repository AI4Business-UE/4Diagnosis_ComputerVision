import './Notifications.css';
import { useNotification } from './NotificationContext';

export default function NotificationContainer() {
  const { notifications } = useNotification();

  return (
    <div className="notification-container">
      {notifications.map(notification => (
        <div key={notification.id} className={`notification notification-${notification.type}`}>
          <div className="notification-content">
            {notification.type === 'loading' && (
              <div className="notification-spinner"></div>
            )}
            {notification.type === 'success' && (
              <span className="notification-icon">✓</span>
            )}
            {notification.type === 'error' && (
              <span className="notification-icon">✕</span>
            )}
            {notification.type === 'info' && (
              <span className="notification-icon">ℹ</span>
            )}
            <span className="notification-message">{notification.message}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
