-- Portal MVP schema bootstrap for SQL Server.
-- Use this when migrations are not yet generated in the environment.

IF OBJECT_ID('dbo.runs', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.runs (
        run_uid NVARCHAR(64) NOT NULL PRIMARY KEY,
        site_id NVARCHAR(100) NOT NULL,
        camera_id NVARCHAR(100) NOT NULL,
        started_at_utc DATETIMEOFFSET NULL,
        ended_at_utc DATETIMEOFFSET NULL,
        source_type NVARCHAR(40) NULL,
        source_value NVARCHAR(400) NULL,
        model_version NVARCHAR(200) NULL,
        cfg_version NVARCHAR(200) NULL,
        line_mode NVARCHAR(40) NULL,
        line_id NVARCHAR(120) NULL,
        fps FLOAT NULL,
        frame_width INT NULL,
        frame_height INT NULL,
        health_summary_json NVARCHAR(MAX) NULL,
        report_csv_relpath NVARCHAR(260) NULL,
        updated_at_utc DATETIMEOFFSET NOT NULL
    );

    CREATE INDEX ix_runs_site_camera_start ON dbo.runs(site_id, camera_id, started_at_utc);
END;
GO

IF OBJECT_ID('dbo.events', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.events (
        event_uid NVARCHAR(64) NOT NULL PRIMARY KEY,
        run_uid NVARCHAR(64) NOT NULL,
        site_id NVARCHAR(100) NOT NULL,
        camera_id NVARCHAR(100) NOT NULL,
        occurred_at_utc DATETIMEOFFSET NULL,
        frame_index INT NULL,
        video_time_s FLOAT NULL,
        direction NVARCHAR(16) NULL,
        track_id INT NULL,
        class_id INT NULL,
        class_name NVARCHAR(80) NULL,
        confidence FLOAT NULL,
        bbox_json NVARCHAR(120) NULL,
        line_mode NVARCHAR(40) NULL,
        occurred_at_utc_source NVARCHAR(40) NULL,
        thumb_path NVARCHAR(260) NULL,
        scene_path NVARCHAR(260) NULL,
        updated_at_utc DATETIMEOFFSET NOT NULL,
        CONSTRAINT fk_events_runs FOREIGN KEY (run_uid) REFERENCES dbo.runs(run_uid) ON DELETE CASCADE
    );

    CREATE INDEX ix_events_site_camera_time ON dbo.events(site_id, camera_id, occurred_at_utc);
    CREATE INDEX ix_events_direction ON dbo.events(direction);
    CREATE INDEX ix_events_class_name ON dbo.events(class_name);
END;
GO

IF OBJECT_ID('dbo.event_reviews', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.event_reviews (
        event_uid NVARCHAR(64) NOT NULL PRIMARY KEY,
        review_status NVARCHAR(20) NOT NULL,
        reviewed_at_utc DATETIMEOFFSET NULL,
        reviewed_by NVARCHAR(120) NULL,
        notes NVARCHAR(MAX) NULL,
        updated_at_utc DATETIMEOFFSET NOT NULL,
        CONSTRAINT fk_event_reviews_events FOREIGN KEY (event_uid) REFERENCES dbo.events(event_uid) ON DELETE CASCADE
    );

    CREATE INDEX ix_event_reviews_status ON dbo.event_reviews(review_status);
END;
GO

IF OBJECT_ID('dbo.camera_criteria', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.camera_criteria (
        site_id NVARCHAR(100) NOT NULL,
        camera_id NVARCHAR(100) NOT NULL,
        criteria_title NVARCHAR(200) NOT NULL,
        criteria_description NVARCHAR(MAX) NOT NULL,
        updated_at_utc DATETIMEOFFSET NOT NULL,
        CONSTRAINT pk_camera_criteria PRIMARY KEY (site_id, camera_id)
    );
END;
GO
