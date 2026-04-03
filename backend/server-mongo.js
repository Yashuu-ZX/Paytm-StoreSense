/**
 * StoreSense Phase 5: Production Backend with MongoDB
 * ====================================================
 * 
 * Production-ready Express server with:
 * - MongoDB persistence via Mongoose
 * - Bulk event ingestion for offline sync
 * - Aggregation pipelines for analytics
 * - Graceful error handling
 * 
 * Environment Variables:
 *   MONGODB_URI - MongoDB connection string (required)
 *   PORT - Server port (default: 3001)
 *   NODE_ENV - Environment (development/production)
 * 
 * @author StoreSense Team
 * @version 5.0 (Production)
 */

require('dotenv').config();

const express = require('express');
const cors = require('cors');
const morgan = require('morgan');
const mongoose = require('mongoose');

// =============================================================================
// CONFIGURATION
// =============================================================================

const config = {
    port: process.env.PORT || 3001,
    mongoUri: process.env.MONGODB_URI || 'mongodb://localhost:27017/storesense',
    nodeEnv: process.env.NODE_ENV || 'development',
    
    // Analytics thresholds
    thresholds: {
        coldZoneNeglectPct: 30,      // Zone is "cold" if neglect > 30%
        trafficTrapPutbackPct: 40,   // Zone is "traffic trap" if put-back ratio > 40%
        hotZoneTakenCount: 20,       // Zone is "hot" if taken > 20 in period
    }
};

// =============================================================================
// MONGOOSE SCHEMAS & MODELS
// =============================================================================

/**
 * TelemetryEvent Schema
 * Stores individual telemetry events from edge devices
 */
const telemetryEventSchema = new mongoose.Schema({
    timestamp: {
        type: Date,
        required: true,
        index: true
    },
    zone_id: {
        type: String,
        required: true,
        index: true
    },
    event_type: {
        type: String,
        required: true,
        enum: ['TAKEN', 'PUT_BACK', 'TOUCH', 'STATUS'],
        index: true
    },
    neglect_rate_pct: {
        type: Number,
        default: 0,
        min: 0,
        max: 100
    },
    // Metadata
    store_id: {
        type: String,
        default: 'default',
        index: true
    },
    device_id: {
        type: String,
        default: 'edge-001'
    }
}, {
    timestamps: true,  // Adds createdAt and updatedAt
    collection: 'telemetry_events'
});

// Compound indexes for common queries
telemetryEventSchema.index({ zone_id: 1, timestamp: -1 });
telemetryEventSchema.index({ store_id: 1, timestamp: -1 });
telemetryEventSchema.index({ event_type: 1, timestamp: -1 });

const TelemetryEvent = mongoose.model('TelemetryEvent', telemetryEventSchema);

/**
 * ZoneSummary Schema
 * Cached aggregated statistics per zone (updated periodically)
 */
const zoneSummarySchema = new mongoose.Schema({
    zone_id: {
        type: String,
        required: true,
        unique: true
    },
    store_id: {
        type: String,
        default: 'default'
    },
    total_interactions: {
        type: Number,
        default: 0
    },
    total_taken: {
        type: Number,
        default: 0
    },
    total_put_back: {
        type: Number,
        default: 0
    },
    total_touch: {
        type: Number,
        default: 0
    },
    latest_neglect_rate: {
        type: Number,
        default: 0
    },
    last_interaction_at: {
        type: Date
    }
}, {
    timestamps: true,
    collection: 'zone_summaries'
});

const ZoneSummary = mongoose.model('ZoneSummary', zoneSummarySchema);

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
app.use(morgan(config.nodeEnv === 'production' ? 'combined' : 'dev'));

// =============================================================================
// API ENDPOINTS
// =============================================================================

/**
 * POST /api/telemetry
 * 
 * Receives telemetry events from edge devices.
 * Supports single event, batch array, and StoreSense telemetry format.
 * 
 * Formats supported:
 * 1. Single event: { timestamp, zone_id, event, neglect_rate_pct }
 * 2. Batch array: { events: [...] }
 * 3. StoreSense: { timestamp, zones: [...] }
 */
app.post('/api/telemetry', async (req, res) => {
    try {
        const { body } = req;
        let events = [];
        
        // Parse different payload formats
        if (body.events && Array.isArray(body.events)) {
            // Batch format from offline queue
            events = body.events.map(e => ({
                timestamp: new Date(e.timestamp * 1000),
                zone_id: e.zone_id,
                event_type: (e.event || e.event_type || 'STATUS').toUpperCase(),
                neglect_rate_pct: e.neglect_rate_pct || 0,
                store_id: e.store_id || 'default',
                device_id: e.device_id || 'edge-001'
            }));
        } else if (body.timestamp && body.zone_id && (body.event || body.event_type)) {
            // Single event format
            events = [{
                timestamp: new Date(body.timestamp * 1000),
                zone_id: body.zone_id,
                event_type: (body.event || body.event_type).toUpperCase(),
                neglect_rate_pct: body.neglect_rate_pct || 0,
                store_id: body.store_id || 'default',
                device_id: body.device_id || 'edge-001'
            }];
        } else if (body.zones && Array.isArray(body.zones)) {
            // StoreSense telemetry format (from Phase 2)
            const baseTimestamp = body.timestamp ? new Date(body.timestamp) : new Date();
            
            events = body.zones.flatMap(zone => {
                const zoneEvents = [];
                
                // Process recent_events if present
                if (zone.recent_events && zone.recent_events.length > 0) {
                    zone.recent_events.forEach(evt => {
                        zoneEvents.push({
                            timestamp: evt.timestamp ? new Date(evt.timestamp) : baseTimestamp,
                            zone_id: zone.zone_id,
                            event_type: (evt.event || 'STATUS').toUpperCase(),
                            neglect_rate_pct: zone.neglect_rate_percent || 0,
                            store_id: body.store_id || 'default',
                            device_id: body.device_id || 'edge-001'
                        });
                    });
                }
                
                // If no recent events, create status update
                if (zoneEvents.length === 0) {
                    zoneEvents.push({
                        timestamp: baseTimestamp,
                        zone_id: zone.zone_id,
                        event_type: 'STATUS',
                        neglect_rate_pct: zone.neglect_rate_percent || 0,
                        store_id: body.store_id || 'default',
                        device_id: body.device_id || 'edge-001'
                    });
                }
                
                return zoneEvents;
            });
        } else {
            return res.status(400).json({
                error: 'Invalid payload format',
                expected: {
                    single: { timestamp: 'unix_seconds', zone_id: 'string', event: 'string' },
                    batch: { events: '[array of single events]' },
                    storesense: { timestamp: 'ISO string', zones: '[array of zone data]' }
                }
            });
        }
        
        // Filter out invalid events
        const validEvents = events.filter(e => e.zone_id && e.event_type);
        
        if (validEvents.length === 0) {
            return res.status(400).json({ error: 'No valid events in payload' });
        }
        
        // Bulk insert events
        const inserted = await TelemetryEvent.insertMany(validEvents, { ordered: false });
        
        // Update zone summaries (async, don't wait)
        updateZoneSummaries(validEvents).catch(err => {
            console.error('[SUMMARY] Update error:', err.message);
        });
        
        console.log(`[TELEMETRY] Processed ${inserted.length} events`);
        
        res.json({
            success: true,
            processed: inserted.length,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('[TELEMETRY] Error:', error);
        
        // Handle duplicate key errors gracefully
        if (error.code === 11000) {
            return res.json({
                success: true,
                processed: 0,
                message: 'Events already exist'
            });
        }
        
        res.status(500).json({ 
            error: 'Internal server error', 
            details: config.nodeEnv === 'development' ? error.message : undefined
        });
    }
});

/**
 * Update zone summaries based on new events
 */
async function updateZoneSummaries(events) {
    const zoneUpdates = {};
    
    // Aggregate updates by zone
    for (const event of events) {
        if (!zoneUpdates[event.zone_id]) {
            zoneUpdates[event.zone_id] = {
                interactions: 0,
                taken: 0,
                put_back: 0,
                touch: 0,
                neglect_rate: event.neglect_rate_pct,
                last_interaction: event.timestamp
            };
        }
        
        const update = zoneUpdates[event.zone_id];
        update.interactions++;
        
        if (event.event_type === 'TAKEN') update.taken++;
        else if (event.event_type === 'PUT_BACK') update.put_back++;
        else if (event.event_type === 'TOUCH') update.touch++;
        
        if (event.timestamp > update.last_interaction) {
            update.last_interaction = event.timestamp;
            update.neglect_rate = event.neglect_rate_pct;
        }
    }
    
    // Apply updates
    for (const [zoneId, update] of Object.entries(zoneUpdates)) {
        await ZoneSummary.findOneAndUpdate(
            { zone_id: zoneId },
            {
                $inc: {
                    total_interactions: update.interactions,
                    total_taken: update.taken,
                    total_put_back: update.put_back,
                    total_touch: update.touch
                },
                $set: {
                    latest_neglect_rate: update.neglect_rate,
                    last_interaction_at: update.last_interaction
                }
            },
            { upsert: true, new: true }
        );
    }
}

/**
 * GET /api/analytics/summary
 * 
 * Returns aggregated analytics for all zones.
 * Uses MongoDB aggregation for real-time statistics.
 */
app.get('/api/analytics/summary', async (req, res) => {
    try {
        const { period = 'today', store_id = 'default' } = req.query;
        
        // Calculate time range
        let startDate = new Date();
        startDate.setHours(0, 0, 0, 0);
        
        if (period === 'week') {
            startDate.setDate(startDate.getDate() - 7);
        } else if (period === 'month') {
            startDate.setMonth(startDate.getMonth() - 1);
        } else if (period === 'all') {
            startDate = new Date(0);
        }
        
        // Aggregation pipeline for zone statistics
        const zoneStats = await TelemetryEvent.aggregate([
            {
                $match: {
                    timestamp: { $gte: startDate },
                    event_type: { $in: ['TAKEN', 'PUT_BACK', 'TOUCH'] }
                }
            },
            {
                $group: {
                    _id: '$zone_id',
                    total_interactions: { $sum: 1 },
                    total_taken: {
                        $sum: { $cond: [{ $eq: ['$event_type', 'TAKEN'] }, 1, 0] }
                    },
                    total_put_back: {
                        $sum: { $cond: [{ $eq: ['$event_type', 'PUT_BACK'] }, 1, 0] }
                    },
                    total_touch: {
                        $sum: { $cond: [{ $eq: ['$event_type', 'TOUCH'] }, 1, 0] }
                    },
                    avg_neglect_rate: { $avg: '$neglect_rate_pct' },
                    last_neglect_rate: { $last: '$neglect_rate_pct' },
                    last_interaction: { $max: '$timestamp' }
                }
            },
            {
                $sort: { _id: 1 }
            }
        ]);
        
        // Process zones and generate alerts
        const processedZones = [];
        const alerts = [];
        
        for (const zone of zoneStats) {
            const totalInteractions = zone.total_interactions || 0;
            const totalTaken = zone.total_taken || 0;
            const totalPutBack = zone.total_put_back || 0;
            const neglectRate = zone.last_neglect_rate || zone.avg_neglect_rate || 0;
            
            // Calculate rates
            const conversionRate = totalInteractions > 0
                ? ((totalTaken / totalInteractions) * 100).toFixed(1)
                : 0;
            const frictionRate = totalInteractions > 0
                ? ((totalPutBack / totalInteractions) * 100).toFixed(1)
                : 0;
            
            // Determine status
            let status = 'NORMAL';
            if (neglectRate > config.thresholds.coldZoneNeglectPct) {
                status = 'COLD';
            } else if (parseFloat(frictionRate) > config.thresholds.trafficTrapPutbackPct) {
                status = 'TRAFFIC_TRAP';
            } else if (totalTaken > config.thresholds.hotZoneTakenCount) {
                status = 'HOT';
            }
            
            const processedZone = {
                zone_id: zone._id,
                total_interactions: totalInteractions,
                total_taken: totalTaken,
                total_put_back: totalPutBack,
                total_touch: zone.total_touch || 0,
                conversion_rate: parseFloat(conversionRate),
                friction_rate: parseFloat(frictionRate),
                neglect_rate: parseFloat(neglectRate.toFixed(1)),
                status,
                last_interaction: zone.last_interaction
            };
            
            processedZones.push(processedZone);
            
            // Generate alerts
            if (status === 'COLD') {
                alerts.push({
                    type: 'COLD_ZONE',
                    severity: 'warning',
                    zone_id: zone._id,
                    message: `${zone._id} has high neglect rate (${neglectRate.toFixed(1)}%). Consider repositioning products.`,
                    metric: neglectRate
                });
            } else if (status === 'TRAFFIC_TRAP') {
                alerts.push({
                    type: 'TRAFFIC_TRAP',
                    severity: 'info',
                    zone_id: zone._id,
                    message: `${zone._id} has high put-back friction (${frictionRate}%). Customers browse but don't buy.`,
                    metric: parseFloat(frictionRate)
                });
            } else if (status === 'HOT') {
                alerts.push({
                    type: 'HOT_ZONE',
                    severity: 'success',
                    zone_id: zone._id,
                    message: `${zone._id} is performing well with ${totalTaken} items taken!`,
                    metric: totalTaken
                });
            }
        }
        
        // Calculate totals
        const summary = {
            total_zones: processedZones.length,
            total_interactions: processedZones.reduce((sum, z) => sum + z.total_interactions, 0),
            total_taken: processedZones.reduce((sum, z) => sum + z.total_taken, 0),
            total_put_back: processedZones.reduce((sum, z) => sum + z.total_put_back, 0),
            avg_neglect_rate: processedZones.length > 0
                ? (processedZones.reduce((sum, z) => sum + z.neglect_rate, 0) / processedZones.length).toFixed(1)
                : 0
        };
        
        res.json({
            timestamp: new Date().toISOString(),
            period,
            summary,
            zones: processedZones,
            alerts
        });
        
    } catch (error) {
        console.error('[ANALYTICS] Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

/**
 * GET /api/analytics/zones/:zoneId
 * 
 * Get detailed analytics for a specific zone.
 */
app.get('/api/analytics/zones/:zoneId', async (req, res) => {
    try {
        const { zoneId } = req.params;
        const limit = parseInt(req.query.limit) || 100;
        
        // Get recent events
        const events = await TelemetryEvent.find({ zone_id: zoneId })
            .sort({ timestamp: -1 })
            .limit(limit)
            .lean();
        
        if (events.length === 0) {
            return res.status(404).json({ error: 'Zone not found or no events' });
        }
        
        // Calculate hourly distribution
        const hourlyDistribution = {};
        events.forEach(evt => {
            const hour = new Date(evt.timestamp).getHours();
            if (!hourlyDistribution[hour]) {
                hourlyDistribution[hour] = { taken: 0, put_back: 0, touch: 0 };
            }
            if (evt.event_type === 'TAKEN') hourlyDistribution[hour].taken++;
            else if (evt.event_type === 'PUT_BACK') hourlyDistribution[hour].put_back++;
            else if (evt.event_type === 'TOUCH') hourlyDistribution[hour].touch++;
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
app.get('/api/analytics/hourly', async (req, res) => {
    try {
        const since = new Date(Date.now() - 24 * 60 * 60 * 1000);
        
        const hourlyStats = await TelemetryEvent.aggregate([
            {
                $match: {
                    timestamp: { $gte: since },
                    event_type: { $in: ['TAKEN', 'PUT_BACK', 'TOUCH'] }
                }
            },
            {
                $group: {
                    _id: {
                        zone_id: '$zone_id',
                        hour: { $hour: '$timestamp' }
                    },
                    taken: {
                        $sum: { $cond: [{ $eq: ['$event_type', 'TAKEN'] }, 1, 0] }
                    },
                    put_back: {
                        $sum: { $cond: [{ $eq: ['$event_type', 'PUT_BACK'] }, 1, 0] }
                    },
                    touch: {
                        $sum: { $cond: [{ $eq: ['$event_type', 'TOUCH'] }, 1, 0] }
                    }
                }
            },
            {
                $sort: { '_id.zone_id': 1, '_id.hour': 1 }
            }
        ]);
        
        // Organize by zone and hour
        const hourlyData = {};
        hourlyStats.forEach(row => {
            const zoneId = row._id.zone_id;
            const hour = row._id.hour.toString().padStart(2, '0');
            
            if (!hourlyData[zoneId]) {
                hourlyData[zoneId] = {};
            }
            hourlyData[zoneId][hour] = {
                taken: row.taken,
                put_back: row.put_back,
                touch: row.touch
            };
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
 * GET /api/events/recent
 * 
 * Get recent telemetry events.
 */
app.get('/api/events/recent', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 50;
        
        const events = await TelemetryEvent.find()
            .sort({ timestamp: -1 })
            .limit(limit)
            .lean();
        
        res.json({
            count: events.length,
            events: events.map(evt => ({
                ...evt,
                timestamp_formatted: evt.timestamp.toISOString()
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
app.delete('/api/events/cleanup', async (req, res) => {
    try {
        const days = parseInt(req.query.days) || 30;
        const cutoff = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
        
        const result = await TelemetryEvent.deleteMany({
            timestamp: { $lt: cutoff }
        });
        
        res.json({
            success: true,
            deleted: result.deletedCount,
            cutoff_date: cutoff.toISOString()
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
app.get('/api/health', async (req, res) => {
    const mongoStatus = mongoose.connection.readyState === 1 ? 'connected' : 'disconnected';
    
    res.json({
        status: mongoStatus === 'connected' ? 'healthy' : 'degraded',
        uptime: process.uptime(),
        timestamp: new Date().toISOString(),
        database: 'mongodb',
        mongodb_status: mongoStatus,
        version: '5.0',
        environment: config.nodeEnv
    });
});

/**
 * GET /api/stats
 * 
 * Get database statistics.
 */
app.get('/api/stats', async (req, res) => {
    try {
        const eventCount = await TelemetryEvent.countDocuments();
        const zoneCount = await ZoneSummary.countDocuments();
        
        const oldestEvent = await TelemetryEvent.findOne()
            .sort({ timestamp: 1 })
            .select('timestamp')
            .lean();
        
        const newestEvent = await TelemetryEvent.findOne()
            .sort({ timestamp: -1 })
            .select('timestamp')
            .lean();
        
        res.json({
            total_events: eventCount,
            total_zones: zoneCount,
            oldest_event: oldestEvent?.timestamp,
            newest_event: newestEvent?.timestamp,
            database: 'mongodb'
        });
        
    } catch (error) {
        console.error('[STATS] Error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
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
// MONGODB CONNECTION & SERVER START
// =============================================================================

async function connectToMongoDB() {
    try {
        console.log('[DB] Connecting to MongoDB...');
        
        await mongoose.connect(config.mongoUri, {
            // Modern Mongoose doesn't need these options anymore
            // but they're here for documentation
        });
        
        console.log('[DB] Connected to MongoDB successfully');
        
        // Create indexes
        await TelemetryEvent.createIndexes();
        await ZoneSummary.createIndexes();
        
        console.log('[DB] Indexes created');
        
    } catch (error) {
        console.error('[DB] MongoDB connection error:', error.message);
        throw error;
    }
}

async function startServer() {
    try {
        // Connect to MongoDB
        await connectToMongoDB();
        
        // Start Express server
        app.listen(config.port, () => {
            console.log('');
            console.log('='.repeat(60));
            console.log('  STORESENSE - Phase 5: Production API Server');
            console.log('='.repeat(60));
            console.log(`  Server running on: http://localhost:${config.port}`);
            console.log(`  Database: MongoDB (${config.mongoUri})`);
            console.log(`  Environment: ${config.nodeEnv}`);
            console.log('');
            console.log('  Endpoints:');
            console.log('    POST /api/telemetry         - Receive telemetry (bulk)');
            console.log('    GET  /api/analytics/summary - Aggregated analytics');
            console.log('    GET  /api/analytics/hourly  - Hourly statistics');
            console.log('    GET  /api/events/recent     - Recent events');
            console.log('    GET  /api/health            - Health check');
            console.log('    GET  /api/stats             - Database stats');
            console.log('='.repeat(60));
            console.log('');
        });
        
    } catch (error) {
        console.error('[FATAL] Failed to start server:', error.message);
        process.exit(1);
    }
}

// Graceful shutdown
process.on('SIGINT', async () => {
    console.log('\n[SERVER] Shutting down...');
    await mongoose.connection.close();
    process.exit(0);
});

process.on('SIGTERM', async () => {
    console.log('\n[SERVER] Shutting down...');
    await mongoose.connection.close();
    process.exit(0);
});

// Start the server
startServer();

module.exports = app;
