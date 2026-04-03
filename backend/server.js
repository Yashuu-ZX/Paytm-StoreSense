/**
 * StoreSense Phase 4: Telemetry API & Analytics Backend
 * ======================================================
 * 
 * Express server that:
 * 1. Receives telemetry from edge Python script via POST /api/telemetry
 * 2. Stores events in SQLite database
 * 3. Provides aggregated analytics via GET /api/analytics/summary
 * 
 * Architecture:
 * - SQLite via sql.js (pure JavaScript, no native compilation needed)
 * - In-memory caching for frequently accessed analytics
 * - CORS enabled for React frontend communication
 * 
 * @author StoreSense Team
 * @version 4.0
 */

const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const initSqlJs = require('sql.js');
const path = require('path');
const fs = require('fs');

// =============================================================================
// CONFIGURATION
// =============================================================================

const PORT = process.env.PORT || 3001;
const DB_PATH = path.join(__dirname, 'storesense.db');
const CONFIG_PATH = path.join(__dirname, '..', 'config.json');

// Analytics thresholds for alerts
const THRESHOLDS = {
    COLD_ZONE_NEGLECT_PCT: 30,      // Zone is "cold" if neglect > 30%
    TRAFFIC_TRAP_PUTBACK_PCT: 40,   // Zone is "traffic trap" if put-back ratio > 40%
    HOT_ZONE_TAKEN_COUNT: 20,       // Zone is "hot" if taken > 20 in period
};

// =============================================================================
// DATABASE INITIALIZATION
// =============================================================================

let db = null;

/**
 * Initialize SQLite database with required tables.
 * Creates tables if they don't exist.
 */
async function initializeDatabase() {
    const SQL = await initSqlJs();
    
    // Try to load existing database
    try {
        if (fs.existsSync(DB_PATH)) {
            const fileBuffer = fs.readFileSync(DB_PATH);
            db = new SQL.Database(fileBuffer);
            console.log('[DB] Loaded existing database');
        } else {
            db = new SQL.Database();
            console.log('[DB] Created new database');
        }
    } catch (err) {
        console.log('[DB] Creating fresh database');
        db = new SQL.Database();
    }
    
    // Create telemetry events table
    db.run(`
        CREATE TABLE IF NOT EXISTS telemetry_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            zone_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            neglect_rate_pct REAL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    `);
    
    // Create zone summary table (for caching/quick lookups)
    db.run(`
        CREATE TABLE IF NOT EXISTS zone_summary (
            zone_id TEXT PRIMARY KEY,
            total_interactions INTEGER DEFAULT 0,
            total_taken INTEGER DEFAULT 0,
            total_put_back INTEGER DEFAULT 0,
            total_touch INTEGER DEFAULT 0,
            latest_neglect_rate REAL DEFAULT 0,
            last_interaction_at DATETIME,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    `);
    
    // Create indexes for faster queries
    db.run(`CREATE INDEX IF NOT EXISTS idx_telemetry_zone ON telemetry_events(zone_id)`);
    db.run(`CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_events(timestamp)`);
    db.run(`CREATE INDEX IF NOT EXISTS idx_telemetry_event ON telemetry_events(event_type)`);
    
    // Save database
    saveDatabase();
    
    console.log('[DB] Database initialized successfully');
    return db;
}

/**
 * Save database to disk
 */
function saveDatabase() {
    if (db) {
        const data = db.export();
        const buffer = Buffer.from(data);
        fs.writeFileSync(DB_PATH, buffer);
    }
}

// Auto-save database periodically
setInterval(saveDatabase, 30000); // Every 30 seconds

// =============================================================================
// EXPRESS APP SETUP
// =============================================================================

const app = express();

// Middleware
app.use(cors({
    origin: ['http://localhost:3000', 'http://localhost:3001', 'http://127.0.0.1:3000'],
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json({ limit: '10mb' }));
app.use(morgan('dev'));

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Convert sql.js result to array of objects
 */
function resultToObjects(result) {
    if (!result || result.length === 0) return [];
    const [res] = result;
    if (!res || !res.columns || !res.values) return [];
    
    return res.values.map(row => {
        const obj = {};
        res.columns.forEach((col, i) => {
            obj[col] = row[i];
        });
        return obj;
    });
}

function getDefaultCalibrationConfig() {
    return {
        version: '3.0',
        rtsp_url: '0',
        calibration_timestamp: new Date().toISOString(),
        global_settings: {
            store_open_time: '00:00',
            store_close_time: '23:59',
            interaction_friction_window: 10,
            decision_window: 5
        },
        rois: []
    };
}

function readCalibrationConfig() {
    if (!fs.existsSync(CONFIG_PATH)) {
        return getDefaultCalibrationConfig();
    }

    const raw = fs.readFileSync(CONFIG_PATH, 'utf8');
    const parsed = JSON.parse(raw);

    return {
        ...getDefaultCalibrationConfig(),
        ...parsed,
        global_settings: {
            ...getDefaultCalibrationConfig().global_settings,
            ...(parsed.global_settings || {})
        },
        rois: Array.isArray(parsed.rois) ? parsed.rois : []
    };
}

function validateCalibrationConfig(config) {
    if (!config || typeof config !== 'object') {
        return 'Config body must be an object';
    }

    if (!config.global_settings || typeof config.global_settings !== 'object') {
        return 'global_settings is required';
    }

    if (!Array.isArray(config.rois)) {
        return 'rois must be an array';
    }

    for (const roi of config.rois) {
        if (!roi.zone_id || typeof roi.zone_id !== 'string') {
            return 'Each ROI must include a zone_id';
        }
        if (![roi.x, roi.y, roi.width, roi.height].every(Number.isFinite)) {
            return 'Each ROI must include numeric x, y, width, and height';
        }
        if (roi.tripwire) {
            if (!Array.isArray(roi.tripwire) || roi.tripwire.length !== 2) {
                return 'Each ROI tripwire must contain exactly 2 points';
            }
        }
        if (roi.shelf_side && !['left', 'right'].includes(roi.shelf_side)) {
            return 'shelf_side must be left or right';
        }
    }

    return null;
}

// =============================================================================
// API ENDPOINTS
// =============================================================================

app.get('/api/calibration-config', (req, res) => {
    try {
        res.json(readCalibrationConfig());
    } catch (error) {
        console.error('[CALIBRATION] Read error:', error);
        res.status(500).json({ error: 'Failed to read calibration config' });
    }
});

app.post('/api/calibration-config', (req, res) => {
    try {
        const config = {
            ...getDefaultCalibrationConfig(),
            ...req.body,
            global_settings: {
                ...getDefaultCalibrationConfig().global_settings,
                ...((req.body && req.body.global_settings) || {})
            },
            calibration_timestamp: new Date().toISOString()
        };

        const validationError = validateCalibrationConfig(config);
        if (validationError) {
            return res.status(400).json({ error: validationError });
        }

        fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2));
        res.json({ success: true, config });
    } catch (error) {
        console.error('[CALIBRATION] Save error:', error);
        res.status(500).json({ error: 'Failed to save calibration config' });
    }
});

/**
 * POST /api/telemetry
 * 
 * Receives telemetry events from the edge Python script.
 */
app.post('/api/telemetry', (req, res) => {
    try {
        if (!db) {
            return res.status(503).json({ error: 'Database not initialized' });
        }
        
        const { body } = req;
        
        // Handle both single event and batch formats
        let events = [];
        
        if (body.events && Array.isArray(body.events)) {
            // Batch format
            events = body.events;
        } else if (body.timestamp && body.zone_id && body.event) {
            // Single event format
            events = [body];
        } else if (body.zones && Array.isArray(body.zones)) {
            // StoreSense telemetry format (from Phase 2)
            events = body.zones.flatMap(zone => {
                const zoneEvents = [];
                // Add recent events from this zone
                if (zone.recent_events) {
                    zone.recent_events.forEach(evt => {
                        zoneEvents.push({
                            timestamp: Math.floor(new Date(evt.timestamp || body.timestamp).getTime() / 1000),
                            zone_id: zone.zone_id,
                            event: evt.event,
                            neglect_rate_pct: zone.neglect_rate_percent || 0
                        });
                    });
                }
                // If no recent events, create a status update
                if (zoneEvents.length === 0) {
                    zoneEvents.push({
                        timestamp: Math.floor(new Date(body.timestamp).getTime() / 1000),
                        zone_id: zone.zone_id,
                        event: 'STATUS',
                        neglect_rate_pct: zone.neglect_rate_percent || 0
                    });
                }
                return zoneEvents;
            });
        } else {
            return res.status(400).json({
                error: 'Invalid payload format',
                expected: {
                    single: { timestamp: 'number', zone_id: 'string', event: 'string', neglect_rate_pct: 'number' },
                    batch: { events: '[array of single events]' },
                    storesense: { timestamp: 'string', zones: '[array of zone data]' }
                }
            });
        }
        
        // Process each event
        const results = [];
        
        for (const event of events) {
            // Validate event
            if (!event.zone_id || !event.event) {
                results.push({ success: false, error: 'Missing zone_id or event' });
                continue;
            }
            
            const timestamp = event.timestamp || Math.floor(Date.now() / 1000);
            const eventType = event.event.toUpperCase();
            const neglectRate = event.neglect_rate_pct || 0;
            
            // Insert into telemetry_events
            db.run(
                `INSERT INTO telemetry_events (timestamp, zone_id, event_type, neglect_rate_pct) VALUES (?, ?, ?, ?)`,
                [timestamp, event.zone_id, eventType, neglectRate]
            );
            
            // Update zone_summary
            const isTaken = eventType === 'TAKEN' ? 1 : 0;
            const isPutBack = eventType === 'PUT_BACK' ? 1 : 0;
            const isTouch = eventType === 'TOUCH' ? 1 : 0;
            
            db.run(`
                INSERT INTO zone_summary (zone_id, total_interactions, total_taken, total_put_back, total_touch, latest_neglect_rate, last_interaction_at, updated_at)
                VALUES (?, 1, ?, ?, ?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(zone_id) DO UPDATE SET
                    total_interactions = total_interactions + 1,
                    total_taken = total_taken + excluded.total_taken,
                    total_put_back = total_put_back + excluded.total_put_back,
                    total_touch = total_touch + excluded.total_touch,
                    latest_neglect_rate = excluded.latest_neglect_rate,
                    last_interaction_at = datetime('now'),
                    updated_at = datetime('now')
            `, [event.zone_id, isTaken, isPutBack, isTouch, neglectRate]);
            
            results.push({ success: true, zone_id: event.zone_id, event: eventType });
        }
        
        // Save database after batch insert
        saveDatabase();
        
        console.log(`[TELEMETRY] Processed ${results.length} events`);
        
        res.json({
            success: true,
            processed: results.length,
            results
        });
        
    } catch (error) {
        console.error('[TELEMETRY] Error:', error);
        res.status(500).json({ error: 'Internal server error', details: error.message });
    }
});

/**
 * GET /api/analytics/summary
 * 
 * Returns aggregated analytics for all zones.
 */
app.get('/api/analytics/summary', (req, res) => {
    try {
        if (!db) {
            return res.status(503).json({ error: 'Database not initialized' });
        }
        
        const result = db.exec(`SELECT * FROM zone_summary ORDER BY zone_id`);
        const zones = resultToObjects(result);
        
        const processedZones = zones.map(zone => {
            const totalInteractions = zone.total_interactions || 0;
            const totalTaken = zone.total_taken || 0;
            const totalPutBack = zone.total_put_back || 0;
            const totalTouch = zone.total_touch || 0;
            const neglectRate = zone.latest_neglect_rate || 0;
            
            // Calculate rates
            const conversionRate = totalInteractions > 0 
                ? ((totalTaken / totalInteractions) * 100).toFixed(1) 
                : 0;
            const frictionRate = totalInteractions > 0 
                ? ((totalPutBack / totalInteractions) * 100).toFixed(1) 
                : 0;
            
            // Determine zone status
            let status = 'NORMAL';
            if (neglectRate > THRESHOLDS.COLD_ZONE_NEGLECT_PCT) {
                status = 'COLD';
            } else if (parseFloat(frictionRate) > THRESHOLDS.TRAFFIC_TRAP_PUTBACK_PCT) {
                status = 'TRAFFIC_TRAP';
            } else if (totalTaken > THRESHOLDS.HOT_ZONE_TAKEN_COUNT) {
                status = 'HOT';
            }
            
            return {
                zone_id: zone.zone_id,
                total_interactions: totalInteractions,
                total_taken: totalTaken,
                total_put_back: totalPutBack,
                total_touch: totalTouch,
                conversion_rate: parseFloat(conversionRate),
                friction_rate: parseFloat(frictionRate),
                neglect_rate: neglectRate,
                status,
                last_interaction: zone.last_interaction_at
            };
        });
        
        // Generate alerts
        const alerts = [];
        processedZones.forEach(zone => {
            if (zone.status === 'COLD') {
                alerts.push({
                    type: 'COLD_ZONE',
                    severity: 'warning',
                    zone_id: zone.zone_id,
                    message: `${zone.zone_id} has high neglect rate (${zone.neglect_rate.toFixed(1)}%). Consider repositioning products.`,
                    metric: zone.neglect_rate
                });
            }
            if (zone.status === 'TRAFFIC_TRAP') {
                alerts.push({
                    type: 'TRAFFIC_TRAP',
                    severity: 'info',
                    zone_id: zone.zone_id,
                    message: `${zone.zone_id} has high put-back friction (${zone.friction_rate}%). Customers browse but don't buy.`,
                    metric: zone.friction_rate
                });
            }
            if (zone.status === 'HOT') {
                alerts.push({
                    type: 'HOT_ZONE',
                    severity: 'success',
                    zone_id: zone.zone_id,
                    message: `${zone.zone_id} is performing well with ${zone.total_taken} items taken!`,
                    metric: zone.total_taken
                });
            }
        });
        
        res.json({
            timestamp: new Date().toISOString(),
            summary: {
                total_zones: processedZones.length,
                total_interactions: processedZones.reduce((sum, z) => sum + z.total_interactions, 0),
                total_taken: processedZones.reduce((sum, z) => sum + z.total_taken, 0),
                total_put_back: processedZones.reduce((sum, z) => sum + z.total_put_back, 0),
                avg_neglect_rate: processedZones.length > 0 
                    ? (processedZones.reduce((sum, z) => sum + z.neglect_rate, 0) / processedZones.length).toFixed(1)
                    : 0
            },
            zones: processedZones,
            alerts
        });
        
    } catch (error) {
        console.error('[ANALYTICS] Error:', error);
        res.status(500).json({ error: 'Internal server error', details: error.message });
    }
});

/**
 * GET /api/analytics/zones/:zoneId
 * 
 * Get detailed analytics for a specific zone.
 */
app.get('/api/analytics/zones/:zoneId', (req, res) => {
    try {
        if (!db) {
            return res.status(503).json({ error: 'Database not initialized' });
        }
        
        const { zoneId } = req.params;
        const limit = parseInt(req.query.limit) || 100;
        
        const result = db.exec(
            `SELECT * FROM telemetry_events WHERE zone_id = ? ORDER BY timestamp DESC LIMIT ?`,
            [zoneId, limit]
        );
        const events = resultToObjects(result);
        
        if (events.length === 0) {
            return res.status(404).json({ error: 'Zone not found or no events' });
        }
        
        // Calculate hourly distribution
        const hourlyDistribution = {};
        events.forEach(evt => {
            const hour = new Date(evt.timestamp * 1000).getHours();
            if (!hourlyDistribution[hour]) {
                hourlyDistribution[hour] = { taken: 0, put_back: 0, touch: 0 };
            }
            if (evt.event_type === 'TAKEN') hourlyDistribution[hour].taken++;
            if (evt.event_type === 'PUT_BACK') hourlyDistribution[hour].put_back++;
            if (evt.event_type === 'TOUCH') hourlyDistribution[hour].touch++;
        });
        
        res.json({
            zone_id: zoneId,
            total_events: events.length,
            recent_events: events.slice(0, 20),
            hourly_distribution: hourlyDistribution
        });
        
    } catch (error) {
        console.error('[ANALYTICS] Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

/**
 * GET /api/analytics/hourly
 * 
 * Get hourly statistics for all zones (last 24 hours).
 */
app.get('/api/analytics/hourly', (req, res) => {
    try {
        if (!db) {
            return res.status(503).json({ error: 'Database not initialized' });
        }
        
        const since = Math.floor(Date.now() / 1000) - (24 * 60 * 60); // Last 24 hours
        const result = db.exec(`
            SELECT 
                zone_id,
                strftime('%H', datetime(timestamp, 'unixepoch')) as hour,
                event_type,
                COUNT(*) as count
            FROM telemetry_events
            WHERE timestamp > ?
            GROUP BY zone_id, hour, event_type
        `, [since]);
        const stats = resultToObjects(result);
        
        // Organize by zone and hour
        const hourlyData = {};
        stats.forEach(row => {
            if (!hourlyData[row.zone_id]) {
                hourlyData[row.zone_id] = {};
            }
            if (!hourlyData[row.zone_id][row.hour]) {
                hourlyData[row.zone_id][row.hour] = { taken: 0, put_back: 0, touch: 0 };
            }
            if (row.event_type === 'TAKEN') hourlyData[row.zone_id][row.hour].taken = row.count;
            if (row.event_type === 'PUT_BACK') hourlyData[row.zone_id][row.hour].put_back = row.count;
            if (row.event_type === 'TOUCH') hourlyData[row.zone_id][row.hour].touch = row.count;
        });
        
        res.json({
            period: '24h',
            data: hourlyData
        });
        
    } catch (error) {
        console.error('[ANALYTICS] Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

/**
 * POST /api/analytics/reset
 *
 * Clears stored telemetry and cached summaries.
 */
app.post('/api/analytics/reset', (req, res) => {
    try {
        if (!db) {
            return res.status(503).json({ error: 'Database not initialized' });
        }

        db.run(`DELETE FROM telemetry_events`);
        db.run(`DELETE FROM zone_summary`);
        saveDatabase();

        res.json({ success: true, message: 'Analytics history reset' });
    } catch (error) {
        console.error('[ANALYTICS RESET] Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

/**
 * GET /api/events/recent
 * 
 * Get recent telemetry events.
 */
app.get('/api/events/recent', (req, res) => {
    try {
        if (!db) {
            return res.status(503).json({ error: 'Database not initialized' });
        }
        
        const limit = parseInt(req.query.limit) || 50;
        const result = db.exec(`SELECT * FROM telemetry_events ORDER BY timestamp DESC LIMIT ?`, [limit]);
        const events = resultToObjects(result);
        
        res.json({
            count: events.length,
            events: events.map(evt => ({
                ...evt,
                timestamp_formatted: new Date(evt.timestamp * 1000).toISOString()
            }))
        });
        
    } catch (error) {
        console.error('[EVENTS] Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

/**
 * DELETE /api/events/cleanup
 * 
 * Clean up old events (older than specified days).
 */
app.delete('/api/events/cleanup', (req, res) => {
    try {
        if (!db) {
            return res.status(503).json({ error: 'Database not initialized' });
        }
        
        const days = parseInt(req.query.days) || 30;
        const cutoff = Math.floor(Date.now() / 1000) - (days * 24 * 60 * 60);
        
        // Get count before deletion
        const beforeResult = db.exec(`SELECT COUNT(*) as count FROM telemetry_events WHERE timestamp < ?`, [cutoff]);
        const beforeCount = resultToObjects(beforeResult)[0]?.count || 0;
        
        db.run(`DELETE FROM telemetry_events WHERE timestamp < ?`, [cutoff]);
        saveDatabase();
        
        res.json({
            success: true,
            deleted: beforeCount,
            cutoff_date: new Date(cutoff * 1000).toISOString()
        });
        
    } catch (error) {
        console.error('[CLEANUP] Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

/**
 * GET /api/health
 * 
 * Health check endpoint.
 */
app.get('/api/health', (req, res) => {
    res.json({
        status: db ? 'healthy' : 'initializing',
        uptime: process.uptime(),
        timestamp: new Date().toISOString(),
        database: 'sqlite (sql.js)',
        version: '4.0'
    });
});

// =============================================================================
// ERROR HANDLING
// =============================================================================

// 404 handler
app.use((req, res) => {
    res.status(404).json({ error: 'Endpoint not found' });
});

// Global error handler
app.use((err, req, res, next) => {
    console.error('[ERROR]', err);
    res.status(500).json({ error: 'Internal server error' });
});

// =============================================================================
// GRACEFUL SHUTDOWN
// =============================================================================

process.on('SIGINT', () => {
    console.log('\n[SERVER] Shutting down...');
    saveDatabase();
    if (db) db.close();
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n[SERVER] Shutting down...');
    saveDatabase();
    if (db) db.close();
    process.exit(0);
});

// =============================================================================
// START SERVER
// =============================================================================

async function startServer() {
    await initializeDatabase();
    
    app.listen(PORT, () => {
        console.log('');
        console.log('='.repeat(60));
        console.log('  STORESENSE - Phase 4: Telemetry API Server');
        console.log('='.repeat(60));
        console.log(`  Server running on: http://localhost:${PORT}`);
        console.log(`  Database: ${DB_PATH}`);
        console.log('');
        console.log('  Endpoints:');
        console.log('    POST /api/telemetry        - Receive telemetry events');
        console.log('    GET  /api/analytics/summary - Get aggregated analytics');
        console.log('    GET  /api/analytics/hourly  - Get hourly statistics');
        console.log('    GET  /api/events/recent     - Get recent events');
        console.log('    GET  /api/health            - Health check');
        console.log('='.repeat(60));
        console.log('');
    });
}

startServer().catch(err => {
    console.error('[FATAL] Failed to start server:', err);
    process.exit(1);
});

module.exports = app;
