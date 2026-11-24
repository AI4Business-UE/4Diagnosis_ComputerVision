import './Notifications.css';
import { useNotification } from './NotificationContext';

interface NotificationContainerProps {
  activeSampleId?: string | null;
}

export default function NotificationContainer({ activeSampleId = null }: NotificationContainerProps) {
  const { notifications, removeNotification } = useNotification();
  const visibleNotifications = notifications.filter(
    n => !n.sampleId || n.sampleId === activeSampleId
  );

  return (
    <div className="notification-container">
      {visibleNotifications.map(notification => (
        <div
          key={notification.id}
          className={`notification notification-${notification.type}`}
          role={notification.type === 'error' ? 'alert' : 'status'}
        >
          <div className="notification-content">
            {notification.type === 'loading' && (
              <div className="notification-spinner"></div>
            )}
            {notification.type === 'success' && (
              <span className="notification-icon">✓</span>
            )}
            {notification.type === 'error' && (
              <span className="notification-error-tag">BŁĄD</span>
            )}
            {notification.type === 'info' && (
              <span className="notification-icon">ℹ</span>
            )}
            <span className="notification-message">{notification.message}</span>
            {notification.onCancel && (
              <button
                type="button"
                className="notification-cancel-btn"
                onClick={notification.onCancel}
                aria-label="Anuluj"
                title="Anuluj"
              >
                ✕
              </button>
            )}
            {!notification.onCancel && notification.type === 'error' && (
              <button
                type="button"
                className="notification-cancel-btn"
                onClick={() => removeNotification(notification.id)}
                aria-label="Zamknij"
                title="Zamknij"
              >
                ✕
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
